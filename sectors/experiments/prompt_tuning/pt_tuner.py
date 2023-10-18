import torch
import argparse
import pandas as pd
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from sectors.utils.models import create_model
from sectors.utils.trie import Trie


class PromptTune(torch.nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        unique_labels: List = None,
        weights: pd.Series = None,
        soft_prompt: torch.Tensor = None,
    ):
        super().__init__()
        self.args = args
        self.ignore_index = -100
        (
            self.model,
            self.tokenizer,
            self.generation_config,
        ) = self.get_model_tokenizer_config()
        self.embeddings = self.model.base_model.get_input_embeddings()
        self.prompt_encoder = PromptEncoder(
            soft_prompt=soft_prompt,
            args=self.args,
            embeddings=self.embeddings,
            tokenizer=self.tokenizer,
            unique_labels=self.unique_labels,
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[PROMPT]"]})
        self.prompt_token_id = self.tokenizer.get_vocab()["[PROMPT]"]
        self.weights = weights

        if args.head == "lh":
            self.unique_labels = unique_labels
            self.max_tokens_to_generate = 150 if args.dataset == "industries" else 75
            self.trie = self.get_trie()
            self.generation_crossentropy = torch.nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
            self.training_crossentropy = torch.nn.CrossEntropyLoss(reduction="none")

    def get_model_tokenizer_config(self) -> Tuple:
        model, tokenizer, generation_config = create_model(
            self.args.model_name, False, self.args.device, self.args.load_in_8bit, parallelize = self.args.parallelize
        )
        tokenizer.col_token_id = None
        for param in model.parameters():
            param.requires_grad = False
        return model, tokenizer, generation_config

    def get_trie(self) -> Trie:
        return Trie(
            self.tokenizer.bos_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.col_token_id,
            self.tokenizer.pad_token_id,
            sequences=[self.tokenizer.encode(label) for label in self.unique_labels],
        )

    def embed_input(self, queries: torch.Tensor) -> torch.Tensor:
        bz = queries.shape[0]
        queries_for_embedding = queries.clone().to(self.args.device)
        queries_for_embedding[
            (queries == self.prompt_token_id)
        ] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (
            (queries == self.prompt_token_id)
            .nonzero()
            .reshape((bz, self.args.sp_len, 2))[:, :, 1]
        )  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.args.sp_len):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, inputs: str) -> torch.Tensor:
        return torch.LongTensor(
            [self.prompt_token_id] * self.args.sp_len + self.tokenizer.encode(inputs)
        )

    def get_query_label(self, inp: str, lab: str) -> Tuple[torch.Tensor, torch.Tensor]:
        query_ids = self.tokenizer.encode(inp + " ")
        label_ids = self.tokenizer.encode(lab)

        query = torch.LongTensor(
            [self.prompt_token_id] * self.args.sp_len
            + query_ids
            + label_ids
            + [self.tokenizer.eos_token_id]
        )
        label = torch.LongTensor(
            [self.ignore_index] * (self.args.sp_len + len(query_ids))
            + label_ids
            + [self.tokenizer.eos_token_id]
        )

        return query, label

    def evaluate_pass(
        self, inputs: List[str], labels: List[str], trie_search: bool = False
    ):
        prefix_allowed_tokens_fn = (
            lambda batch_id, sent: self.trie.get(batch_id, sent.tolist())
            if trie_search
            else None
        )
        queries = [self.get_query(inp).flip(dims=[0]) for inp in inputs]
        queries_padded = (
            pad_sequence(
                queries,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            .flip(dims=[1])
            .long()
            .to(self.args.device)
        )
        inputs_embeds = self.embed_input(queries_padded)

        labels = [label + self.tokenizer.eos_token for label in labels]
        label_tensors = self.tokenizer.batch_encode_plus(
            labels, return_tensors="pt", padding=True
        )
        label_ids = label_tensors["input_ids"]  # .to(self.args.device)
        label_mask = label_tensors["attention_mask"]  # .to(self.args.device)
        labels = torch.where(
            label_mask == 1,
            label_ids,
            torch.tensor(self.ignore_index),  # , device = label_ids.device)
        )
        attention_mask = queries_padded != self.tokenizer.pad_token_id
        attention_mask.to(self.args.device)

        predictions = self.model.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.half(),
            max_new_tokens=self.max_tokens_to_generate,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=self.args.num_beams,
            generation_config=self.generation_config,
            pad_token_id=self.generation_config.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        if self.args.num_beams == 1 and trie_search == False:
            logits = torch.stack(predictions.scores).permute(1, 0, 2)
            seq_len_logits = logits.size(1)
            seq_len_labels = labels.size(1)
            if seq_len_logits < seq_len_labels:
                padding = torch.full(
                    (logits.size(0), seq_len_labels - seq_len_logits, logits.size(2)),
                    -1e10,
                    device=logits.device,
                )

                logits = torch.cat([logits, padding], dim=1)
            elif seq_len_logits > seq_len_labels:
                logits = logits[:, :seq_len_labels, :]

            loss = self.generation_crossentropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).to(logits.device),
            ).item()
        else:
            loss = 0

        predictions_strings = self.tokenizer.batch_decode(
            predictions.sequences, skip_special_tokens=True
        )

        return predictions_strings, loss

    def get_queries_embeds(
        self, inputs: List[str], labels: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries, labels = zip(
            *[self.get_query_label(inp, lab) for inp, lab in zip(inputs, labels)]
        )
        queries = (
            pad_sequence(
                queries, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            .long()
            .to(self.args.device)
        )
        inputs_embeds = self.embed_input(queries)
        return queries, labels, inputs_embeds

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, logits.size(2))
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.training_crossentropy(shift_logits, shift_labels).reshape(
            (labels.size(0), labels.size(1) - 1)
        )
        mask = loss != 0
        sums = torch.sum(loss * mask, dim=1)
        counts = torch.sum(mask, dim=1).float()
        row_averages = sums / counts
        weighted_average = (class_weights * row_averages).sum()
        return weighted_average

    def forward(self, inputs: List[str], labels: List[str]) -> torch.Tensor:
        class_weights = torch.tensor(
            [sum([self.weights[lab] for lab in label.split("; ")]) for label in labels],
            device=self.args.device,
        )
        queries, labels, inputs_embeds = self.get_queries_embeds(inputs, labels)
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
            .long()
            .to(self.args.device)
            .squeeze(1)
        )
        mask = queries != self.tokenizer.pad_token_id
        output = self.model(
            inputs_embeds=inputs_embeds.to(self.args.device),  # .half() TODO
            attention_mask=mask.to(self.args.device),  # .half() TODO
            labels=labels,
        )
        loss = self.compute_loss(output.logits, labels, class_weights)
        return loss


class PromptEncoder(torch.nn.Module):
    def __init__(
        self,
        soft_prompt: torch.Tensor = None,
        args: argparse.Namespace = None,
        embeddings: torch.nn.Embedding = None,
        tokenizer=None,
        unique_labels: List[str] = None,
    ):
        super().__init__()
        if not soft_prompt:
            soft_prompt = self.random_init(args, embeddings, tokenizer, unique_labels)

        self.soft_prompt = torch.nn.Embedding.from_pretrained(soft_prompt, freeze=False)
        self.to(args.device)

    def forward(self) -> torch.nn.Embedding:
        input_embeds = self.soft_prompt(self.seq_indices)
        return input_embeds

    def random_init(self, args, embeddings, tokenizer, unique_labels):
        self.sp_len = args.sp_len
        self.embeddings = embeddings
        self.embeddings.weight.requires_grad = False
        self.tokenizer = tokenizer
        self.seq_indices = torch.arange(
            0, self.sp_len, dtype=torch.long, device=args.device
        )

        # use embeddings of random tokens to initialize soft prompt
        if args.prompt_init == "random":
            init_tokens = torch.arange(
                1000, 1000 + self.sp_len, dtype=torch.long, device=args.device
            )
        # use embeddings of random label tokens to initialize soft prompt
        elif args.prompt_init == "labels":
            all_tokens = (
                tokenizer.encode(" ".join(unique_labels), return_tensors="pt")
                .squeeze(0)
                .to(args.device)
            )
            if len(all_tokens) < self.sp_len:
                additional_needed = self.sp_len - len(all_tokens)
                additional_tokens = torch.arange(
                    1000, 1000 + additional_needed, dtype=torch.long, device=args.device
                )
                init_tokens = torch.cat([all_tokens, additional_tokens]).to(args.device)
            elif len(all_tokens) > self.sp_len:
                random_indices = torch.randperm(all_tokens.size(0))[: self.sp_len]
                init_tokens = all_tokens[random_indices].to(args.device)
            else:
                init_tokens = all_tokens
        init_embeddings = self.embeddings(init_tokens.to(self.embeddings.weight.device))
        return init_embeddings
