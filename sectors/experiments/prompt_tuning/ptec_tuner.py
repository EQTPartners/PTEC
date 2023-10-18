import torch
import argparse
import pandas as pd
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss
from sectors.utils.models import create_model
from sectors.experiments.prompt_tuning import PromptTune


class PTECTune(PromptTune):
    def __init__(
        self,
        args: argparse.Namespace,
        unique_labels: List[str] = None,
        weights: pd.Series = None,
        soft_prompt: torch.Tensor = None,
    ):
        super().__init__(
            args, unique_labels=unique_labels, weights=weights, soft_prompt=soft_prompt
        )
        self.training_crossentropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.weights, device=self.args.device)
        )

    def get_model_tokenizer_config(self):
        model, tokenizer, generation_config = create_model(
            self.args.model_name,
            False,
            self.args.device,
            self.args.load_in_8bit,
            len(self.unique_labels),
            parallelize = self.args.parallelize
        )
        tokenizer.col_token_id = None
        for name, param in model.named_parameters():
            if name == "score.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model, tokenizer, generation_config

    def forward(
        self, inputs: List[str], labels: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_without_loss(inputs)
        loss = self.training_crossentropy(
            out.logits, labels.to(self.args.device).float()
        )
        return out.logits, loss
    
    def forward_without_loss(self, inputs: List[str]):
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
        attention_mask = queries_padded != self.tokenizer.pad_token_id
        out = self.model(
            inputs_embeds=inputs_embeds.to(self.args.device).half(),  # float()
            attention_mask=attention_mask.to(self.args.device).half(),  # float()
        )
        return out
