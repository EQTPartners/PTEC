import os
from os.path import join
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from datetime import datetime

from sectors.config import DATA_DIR
from sectors.utils.evaluation import sector_report
from sectors.utils.dataset import (
    MultiLabelT2TDataset,
    PromptClassificationHeadDataset,
)
from sectors.utils.models import (
    set_seed,
    ordered_set,
    get_predictions,
)
from sectors.experiments.prompt_tuning import (
    PromptTune,
    PTECTune,
)
from sectors.utils.save_load import get_path, save_model_with_timestamp

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, LambdaLR
from transformers.optimization import Adafactor, AdamW
from transformers.utils import logging


logging.set_verbosity_error()
sigmoid = torch.nn.Sigmoid()


def get_trainer_class(head: str):
    return PromptCHTrainer if (head == "ch") else PromptTrainer


def get_tuner_class(head: str):
    return PTECTune if (head == "ch") else PromptTune


def construct_generation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--sp_len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gradient_update_steps", type=int, default=32)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--augmented",
        type=str,
        default="preprocessed",
        choices=["preprocessed", "augmented"],
    )
    parser.add_argument("--labels", type=str, default="keyword", choices=["keyword", "label"])
    parser.add_argument("--prompt_init", type=str, default="labels")
    parser.add_argument("--head", type=str, default="lh", choices=["lh", "ch"])
    parser.add_argument("--differential_lr", type=bool, default=True)
    parser.add_argument("--ch_lr", type=float, default=0.00001)
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdaFactor", "AdamW"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmup",
        choices=["warmup", "exponential", "linear"],
    )
    # parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nobar", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed for init")

    # only needed for hyperparameter tuning
    parser.add_argument("--init_points", type=int, default=25)
    parser.add_argument("--n_iter", type=int, default=15)
    parser.add_argument("--max_lr", type=float, default=1)
    parser.add_argument("--max_sp_len", type=int, default=200)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=18)
    parser.add_argument("--interrupt_threshold", type=float, default=0.07)
    parser.add_argument("--continue_from", type=str)

    # only needed for reproducing
    parser.add_argument("--path", type=str)
    parser.add_argument("--profile_flops", type=bool, default=False)
    # parser.add_argument("--parallelize", action="store_true")
    args = parser.parse_args()
    args.TRAINER_CLASS = get_trainer_class(args.head)
    args.TUNE_CLASS = get_tuner_class(args.head)
    return args


class PromptTrainer(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        set_seed(args.seed)

        # load data
        self.train_set = self.get_dataset(f"train_{args.augmented}.json", self.args)
        self.dev_set = self.get_dataset("dev_preprocessed.json", self.args)
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size)

        self.unique_labels = ordered_set(self.train_set.labels.columns)
        self.model = self.args.TUNE_CLASS(
            args, self.unique_labels, self.train_set.weights
        )
        self.writer = SummaryWriter(
            f"runs/{args.model_name}_lr{args.lr}_splen{args.sp_len}_epochs{args.epochs}_wd{args.weight_decay}_{args.head}_{datetime.now()}"
        )

    def evaluate_pass(
        self,
        inputs: List[str],
        labels: List[str],
        trie_search: bool,
        split: str,
        epoch_idx: int,
        n_examples: int,
    ):
        predictions_strings, loss = self.model.evaluate_pass(
            inputs, labels, trie_search=trie_search
        )
        trues_array = [
            [1 if label in true_string else 0 for label in self.unique_labels]
            for true_string in labels
        ]
        predictions_array = [
            [1 if label in pred_string else 0 for label in self.unique_labels]
            for pred_string in predictions_strings
        ]
        for inp, lab, pred in zip(inputs, labels, predictions_strings):
            self.writer.add_text(
                f"{split}/generated_trie_search"
                if trie_search
                else f"{split}/generated",
                f"Input: {inp} | Label: {lab} | Prediction: {pred}",
                (epoch_idx * len(self.train_loader)) + (7 * n_examples),
            )
        return trues_array, predictions_array, loss

    def forward_pass(
        self, inputs: List[str], labels: List[str], *args
    ) -> Tuple[List, List, torch.Tensor]:
        loss = self.model(inputs, labels)
        return [], [], loss

    def evaluate_score(
        self,
        epoch_idx: int,
        loader: DataLoader,
        split: str,
        trie_search: bool = False,
    ) -> Dict:
        self.model.eval()
        with torch.no_grad():
            trues = []
            preds = []
            tot_loss = 0
            n_examples = 0
            for inputs, labels in tqdm(loader, disable=self.args.nobar):
                n_examples += len(inputs)
                trues_array, predictions_array, loss = self.evaluate_pass(
                    inputs, labels, trie_search, split, epoch_idx, n_examples
                )
                trues += trues_array
                preds += predictions_array
                tot_loss += loss

            report = sector_report(trues, preds, self.unique_labels)
            if loss:
                loss = tot_loss / len(loader)
                self.writer.add_scalar(
                    f"{split}/Loss",
                    loss,
                    epoch_idx * len(self.train_loader) * self.args.batch_size,
                )
            self.writer.add_scalar(
                f"{split}_trie_search/Macro-F1" if trie_search else f"{split}/Macro-F1",
                report["sectors"]["macro avg"]["f1-score"],
                epoch_idx * len(self.train_loader) * self.args.batch_size,
            )
            self.writer.add_scalar(
                f"{split}_trie_search/avg_n_predicted"
                if trie_search
                else f"{split}/avg_n_predicted",
                report["avg_n_predicted"],
                epoch_idx * len(self.train_loader) * self.args.batch_size,
            )
        return report

    def test_evaluation(
        self,
        epoch_idx: int,
        loader: DataLoader,
        split: str,
        trie_search: bool = False,
        factor: int = 40,
    ) -> None:
        self.model.eval()
        with torch.no_grad():
            n_examples = 0
            for idx, (inputs, labels) in tqdm(
                enumerate(loader), disable=self.args.nobar
            ):
                if idx == len(loader) // factor:
                    return None
                n_examples += len(inputs)
                self.evaluate_pass(
                    inputs, labels, trie_search, split, epoch_idx, n_examples
                )
        return NotImplementedError("Model should have already stopped running here")

    def evaluate_loss(
        self, epoch_idx: int, loader: DataLoader, split: str, *args
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            n_examples = 0
            accumulated_loss = 0
            preds = []
            trues = []
            total_loss = 0
            for inputs, labels in tqdm(loader, disable=self.args.nobar):
                n_examples += len(inputs)
                true, pred, loss = self.forward_pass(
                    inputs, labels, split, epoch_idx, n_examples
                )
                preds += pred
                trues += true
                total_loss += loss.item()
                accumulated_loss += loss.item()

                if n_examples % self.args.gradient_update_steps == 0:
                    step = epoch_idx * len(loader) * self.args.batch_size + n_examples
                    self.writer.add_scalar(
                        f"{split}/Loss",
                        accumulated_loss,
                        step,
                    )
                    self.writer.add_scalar(
                        f"{split}/Avg_n_predicted",
                        np.sum(np.array(pred)) / len(inputs),
                        step,
                    )
                    accumulated_loss = 0
            return total_loss

    def save_results(self, direk: str, best_ckpt: Dict):
        self.data = {
            k: best_ckpt[k]
            for k in best_ckpt.keys()
            if k not in ["softprompt", "classification_head"]
        }
        self.data["args"].pop("TRAINER_CLASS")
        self.data["args"].pop("TUNE_CLASS")
        os.makedirs(direk, exist_ok=True)
        with open(join(direk, "results.json"), "w") as f:
            json.dump(self.data, f, indent=4)

    def save(self, best_ckpt: Dict):
        UNCONSTRAINED_PATH = get_path(
            self.args.dataset,
            self.args.augmented,
            "PT",
            "unconstrained",
            self.args.model_name,
        )
        self.save_results(UNCONSTRAINED_PATH, best_ckpt)
        save_model_with_timestamp(
            best_ckpt["softprompt"], UNCONSTRAINED_PATH, "softprompt_"
        )

    def get_params(self) -> List[torch.nn.Parameter]:
        return [self.model.prompt_encoder.soft_prompt.weight]

    def group_params_lr(
        self, params: List[torch.nn.Parameter]
    ) -> Tuple[List[torch.nn.Parameter], float]:
        params_grouped = params
        max_lr = self.args.lr
        return params_grouped, max_lr

    def get_scheduler(self, optimizer, max_lr):
        updates_per_epoch = len(self.train_set) // self.args.gradient_update_steps
        num_updates = self.args.epochs * updates_per_epoch
        if self.args.scheduler == "warmup":
            scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr,
                pct_start=self.args.pct_start,
                total_steps=num_updates,
                cycle_momentum=False,
            )
        elif self.args.scheduler == "exponential":
            decay_rate = 0.01 ** (1 / (num_updates - 1))
            scheduler = ExponentialLR(
                optimizer=optimizer,
                gamma=decay_rate,
            )
        elif self.args.scheduler == "linear":
            lr_lambda = lambda epoch: 1 - epoch / num_updates
            scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

    def get_optimizer(self, params_grouped):
        if self.args.optimizer == "AdaFactor":
            optimizer = Adafactor(
                params_grouped,
                lr=self.args.lr,
                clip_threshold=1.0,
                weight_decay=self.args.weight_decay,
                relative_step=False,
                scale_parameter=False,
            )
        elif self.args.optimizer == "AdamW":
            optimizer = AdamW(
                params_grouped,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                no_deprecation_warning=True,
            )
        return optimizer

    def train(self) -> Dict:
        self.best_dev, self.early_stop = 0, 0
        self.best_ckpt = None
        params = self.get_params()
        params_grouped, max_lr = self.group_params_lr(params)
        optimizer = self.get_optimizer(params_grouped)
        scheduler = self.get_scheduler(optimizer, max_lr)

        for epoch_idx in range(self.args.epochs):
            # run training
            if self.args.augmented == "augmented":
                epoch_idx = epoch_idx * 7
            n_examples = 0
            accumulated_loss = 0
            preds = []
            trues = []
            self.model.train()
            for inputs, labels in tqdm(self.train_loader, disable=self.args.nobar):
                n_examples += len(inputs)
                true, pred, loss = self.forward_pass(
                    inputs, labels, "train", epoch_idx, n_examples
                )
                preds += pred
                trues += true
                accumulated_loss += loss.item()
                loss.backward()

                if n_examples % self.args.gradient_update_steps == 0:
                    step = (
                        epoch_idx * len(self.train_loader) * self.args.batch_size
                        + n_examples
                    )
                    self.writer.add_scalar(
                        "Train/gradient_norm",
                        torch.norm(params[0].grad.detach(), 2),
                        step,
                    )
                    self.writer.add_scalar(
                        "Train/gradient_max",
                        params[0].grad.detach().abs().max(),
                        step,
                    )
                    self.writer.add_scalar(
                        "Train/Loss",
                        accumulated_loss,
                        step,
                    )
                    self.writer.add_scalar(
                        "Train/Avg_n_predicted",
                        np.sum(np.array(pred)) / len(inputs),
                        step,
                    )
                    self.writer.add_scalar(
                        "LR",
                        scheduler.get_last_lr()[0],
                        step,
                    )

                    if self.args.optimizer == "AdamW":
                        clip_grad_norm_(params, 1)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    accumulated_loss = 0
                if (self.args.augmented == "augmented") and ((n_examples % 4096) == 0):
                    epoch_idx += 1
                    if self.evaluate_dev(epoch_idx):
                        if self.best_ckpt:
                            return self.best_ckpt["val_report"]
                        else:
                            return {"sectors": {"macro avg": {"f1-score": 0}}}

            if preds:
                train_report = sector_report(trues, preds, self.unique_labels)
                self.writer.add_scalar(
                    "Train/Macro-F1",
                    train_report["sectors"]["macro avg"]["f1-score"],
                    epoch_idx * len(self.train_loader) * self.args.batch_size,
                )
                self.writer.add_scalar(
                    "Train/avg_n_predicted",
                    train_report["avg_n_predicted"],
                    epoch_idx * len(self.train_loader) * self.args.batch_size,
                )

            if self.evaluate_dev(epoch_idx):
                if self.best_ckpt:
                    return self.best_ckpt["val_report"]
                else:
                    return {"sectors": {"macro avg": {"f1-score": 0}}}
        return self.best_ckpt["val_report"]

    def test_run(self, factor: int):
        params = self.get_params()
        params_grouped, max_lr = self.group_params_lr(params)
        optimizer = self.get_optimizer(params_grouped)
        scheduler = self.get_scheduler(optimizer, max_lr)
        self.model.train()
        n_examples = 0
        for ind, (inputs, labels) in tqdm(
            enumerate(self.train_loader), disable=self.args.nobar
        ):
            if ind == len(self.train_loader) // factor:
                return None
            n_examples += len(inputs)
            _, _, loss = self.forward_pass(inputs, labels, "train", 0, n_examples)
            loss.backward()

            if n_examples % self.args.gradient_update_steps == 0:
                if self.args.optimizer == "AdamW":
                    clip_grad_norm_(params, 1)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        raise NotImplementedError("test run should have stopped before this point")

    def evaluate_dev(self, epoch_idx: int) -> float:
        val_report = self.evaluate_score(epoch_idx, self.dev_loader, "Dev", True)
        if val_report["sectors"]["macro avg"]["f1-score"] > (self.best_dev + 0.005):
            self.best_ckpt = self.get_ckpt(val_report)
            self.early_stop = 0
            self.best_dev = val_report["sectors"]["macro avg"]["f1-score"]
        else:
            self.early_stop += 1
            if (self.early_stop >= self.args.early_stop) or (
                val_report["sectors"]["macro avg"]["f1-score"]
                <= self.args.interrupt_threshold
            ):
                return True
        if (
            val_report["sectors"]["macro avg"]["f1-score"]
            < self.args.interrupt_threshold
        ):
            return True
        if self.args.head == "lh":
            self.evaluate_loss(epoch_idx, self.dev_loader, "Dev", True)
        return False

    def get_ckpt(self, val_report: Dict) -> Dict:
        return {
            "softprompt": self.model.prompt_encoder.state_dict(),
            "val_report": val_report,
            "args": vars(self.args),
        }

    def get_dataset(self, file: str, args: argparse.Namespace) -> MultiLabelT2TDataset:
        return MultiLabelT2TDataset(file, args)

    def load_checkpoint(self, checkpoint: Dict):
        self.model.prompt_encoder.load_state_dict(checkpoint["softprompt"])

    def evaluate_test(self, checkpoint: Dict, trie_search: bool = True) -> Dict:
        self.load_checkpoint(checkpoint)
        TEST_PATH = join(DATA_DIR, self.args.dataset, "test_preprocessed.json")
        self.test_set = self.get_dataset(TEST_PATH, self.args)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size)
        return self.evaluate_score(0, self.test_loader, "Test", trie_search)

    def collect_predictions(
        self,
        loader: DataLoader,
        trie_search: bool,
    ) -> Dict:
        self.model.eval()
        with torch.no_grad():
            trues = []
            preds = []
            for inputs, labels in tqdm(loader):
                trues_array, predictions_array, _ = self.evaluate_pass(
                    inputs, labels, trie_search, "Test", 0, 0
                )
                trues += trues_array
                preds += predictions_array
        return trues, preds


class PromptCHTrainer(PromptTrainer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def get_dataset(
        self, path: str, args: argparse.Namespace
    ) -> PromptClassificationHeadDataset:
        return PromptClassificationHeadDataset(path, args)

    def any_pass(self, inputs: List[str], labels: List[str], *args) -> Tuple:
        logits, loss = self.model(inputs, labels)
        return labels.tolist(), get_predictions(logits), loss

    def prob_pass(self, inputs: List[str], labels: List[str], *args) -> Tuple:
        logits, _ = self.model(inputs, labels)
        return labels.tolist(), sigmoid(logits).tolist()

    def evaluate_pass(self, inputs: List[str], labels: List[str], *args) -> Tuple:
        return self.any_pass(inputs, labels, *args)

    def forward_pass(self, inputs: List[str], labels: List[str], *args) -> Tuple:
        return self.any_pass(inputs, labels, *args)

    def save(self, best_ckpt: Dict):
        UNCONSTRAINED_PATH = get_path(
            self.args.dataset,
            self.args.augmented,
            "PTEC",
            "unconstrained",
            self.args.model_name,
        )
        self.save_results(UNCONSTRAINED_PATH, best_ckpt)
        save_model_with_timestamp(
            best_ckpt["softprompt"], UNCONSTRAINED_PATH, "softprompt_"
        )
        save_model_with_timestamp(
            best_ckpt["classification_head"], UNCONSTRAINED_PATH, "classification_head_"
        )

    def get_params(self) -> List[torch.nn.Parameter]:
        return [
            self.model.prompt_encoder.soft_prompt.weight,
            self.model.model.score.weight,
        ]

    def group_params_lr(self, params):
        if self.args.differential_lr:
            params_grouped = [
                {
                    "params": self.model.prompt_encoder.soft_prompt.weight,
                    "lr": self.args.lr,
                },
                {"params": self.model.model.score.weight, "lr": self.args.ch_lr},
            ]
            max_lr = [self.args.lr, self.args.ch_lr]
            return params_grouped, max_lr
        else:
            params_grouped = params
            max_lr = self.args.lr
            return params_grouped, max_lr

    def get_ckpt(self, val_report: Dict) -> Dict:
        return {
            "softprompt": self.model.prompt_encoder.state_dict(),
            "classification_head": self.model.model.score.state_dict(),
            "val_report": val_report,
            "args": vars(self.args),
        }

    def load_checkpoint(self, checkpoint: Dict):
        self.model.prompt_encoder.load_state_dict(checkpoint["softprompt"])
        self.model.model.score.load_state_dict(checkpoint["classification_head"])

    def collect_probabilities(
        self,
        loader: DataLoader,
    ) -> Dict:
        self.model.eval()
        with torch.no_grad():
            trues = []
            preds = []
            for inputs, labels in tqdm(loader):
                trues_array, predictions_array = self.prob_pass(inputs, labels)
                trues += trues_array
                preds += predictions_array
        return trues, preds


def main():
    args = construct_generation_args()
    print(args)
    trainer = args.TRAINER_CLASS(args)
    trainer.train()
    trainer.best_ckpt["test_report"] = trainer.evaluate_test(trainer.best_ckpt)
    trainer.save(trainer.best_ckpt)


if __name__ == "__main__":
    main()
