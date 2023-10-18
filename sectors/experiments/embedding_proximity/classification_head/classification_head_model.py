import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sectors.utils.evaluation import sector_report
from sectors.utils.dataset import load_embedding_data
from sectors.utils.models import get_predictions
from sectors.utils.save_load import (
    get_path,
    save_model_with_timestamp,
    open_most_recent_model,
)
from sectors.config import TENSORBOARD_DIR


sigmoid = torch.nn.Sigmoid()


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x.float())
        return x
    

class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def evaluate(model, dataloader, labels, display):
    preds = []
    trues = []
    for embeddings_batch, labels_batch in dataloader:
        with torch.no_grad():
            model.eval()
            embeddings_batch = embeddings_batch.to(torch.float32)
            out = model(embeddings_batch)
            preds += get_predictions(out)
            trues += labels_batch.tolist()
    return sector_report(trues, preds, labels, display=display)


def train_one_epoch(
    classificationhead, optimizer, criterion, dataloader, writer, epoch
):
    classificationhead.train()
    preds = []
    trues = []

    for index, batch in enumerate(dataloader):
        embeddings_batch, labels_batch = batch
        embeddings_batch = embeddings_batch.to(torch.float32)
        out = classificationhead(embeddings_batch)
        loss = criterion(out, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classificationhead.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        preds += get_predictions(out)
        trues += labels_batch.tolist()
        if writer:
            writer.add_scalar(
                "train/loss", loss.item(), epoch * len(dataloader) + index
            )

    return preds, trues


def validate(classificationhead, criterion, dataloader, writer, epoch, probas=False):
    classificationhead.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            embeddings_batch, labels_batch = batch
            embeddings_batch = embeddings_batch.to(torch.float32)
            out = classificationhead(embeddings_batch)
            if probas:
                preds += sigmoid(out).tolist()
            else:
                preds += get_predictions(out)
            trues += labels_batch.tolist()
            if writer:
                loss = criterion(out, labels_batch)
                writer.add_scalar(
                    "val/loss", loss.item(), epoch * len(dataloader) + index
                )

    return trues, preds


def get_probs(classificationhead, dataloader):
    return validate(classificationhead, None, dataloader, None, None, True)


def get_dataloader(dataset, split, embedding_type, batch_size, model_name, augmented):
    embeds, labels, columns = load_embedding_data(
        dataset, split, model_name, augmented, embedding_type
    )
    data = CustomDataset(embeds, labels)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader, columns, embeds.size(1)


def train(
    dataset,
    lr,
    weight_decay,
    batch_size,
    n_epochs,
    embedding_type,
    model_name,
    augmented,
):
    train_dataloader, labels, embedding_dim = get_dataloader(
        dataset, "train", embedding_type, batch_size, model_name, augmented
    )
    val_dataloader, labels, embedding_dim = get_dataloader(
        dataset, "dev", embedding_type, batch_size, model_name, augmented
    )

    classificationhead = ClassificationHead(embedding_dim, len(labels))
    optimizer = torch.optim.Adam(
        classificationhead.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_macro_f1 = 0

    for epoch in range(n_epochs):
        _ = train_one_epoch(
            classificationhead, optimizer, criterion, train_dataloader, None, epoch
        )
        trues_val, preds_val = validate(
            classificationhead, None, val_dataloader, None, None
        )
        class_report_val = sector_report(trues_val, preds_val, labels)
        if class_report_val["sectors"]["macro avg"]["f1-score"] > best_val_macro_f1:
            best_val_macro_f1 = class_report_val["sectors"]["macro avg"]["f1-score"]
    return best_val_macro_f1


def train_final(
    dataset,
    lr,
    weight_decay,
    batch_size,
    n_epochs,
    embedding_type,
    model_name,
    augmented,
    flops,
):
    PATH = get_path(
        dataset, augmented, "classification_head", "unconstrained", model_name
    )
    train_dataloader, labels, embedding_dim = get_dataloader(
        dataset, "train", embedding_type, batch_size, model_name, augmented
    )
    val_dataloader, labels, embedding_dim = get_dataloader(
        dataset, "dev", embedding_type, batch_size, model_name, augmented
    )

    classificationhead = ClassificationHead(embedding_dim, len(labels))
    optimizer = torch.optim.Adam(
        classificationhead.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    writer = SummaryWriter(
        os.path.join(
            TENSORBOARD_DIR,
            PATH,
            f"lr{round(lr, 5)}_wd{round(weight_decay, 5)}_{datetime.now()}",
        )
    )
    best_val_macro_f1 = 0

    for epoch in range(n_epochs):
        if flops and (epoch == 0):
            with torch.profiler.profile(with_flops=True) as training_profiler:
                trues_train, preds_train = train_one_epoch(
                    classificationhead,
                    optimizer,
                    criterion,
                    train_dataloader,
                    writer,
                    epoch,
                )
            flops = sum([event.flops for event in training_profiler.key_averages()])
        else:
            trues_train, preds_train = train_one_epoch(
                classificationhead,
                optimizer,
                criterion,
                train_dataloader,
                writer,
                epoch,
            )
        class_report_train = sector_report(trues_train, preds_train, labels)
        writer.add_scalar(
            "train/macro-f1",
            class_report_train["sectors"]["macro avg"]["f1-score"],
            epoch,
        )
        writer.add_scalar(
            "train/micro-f1",
            class_report_train["sectors"]["micro avg"]["f1-score"],
            epoch,
        )
        writer.add_scalar(
            "train/macro-jaccard", class_report_train["jaccard_macro"], epoch
        )
        writer.add_scalar(
            "train/micro-jaccard", class_report_train["jaccard_micro"], epoch
        )

        trues_val, preds_val = validate(
            classificationhead, criterion, val_dataloader, writer, epoch
        )
        class_report_val = sector_report(trues_val, preds_val, labels)
        writer.add_scalar(
            "val/macro-f1", class_report_val["sectors"]["macro avg"]["f1-score"], epoch
        )
        writer.add_scalar(
            "val/micro-f1", class_report_val["sectors"]["micro avg"]["f1-score"], epoch
        )
        writer.add_scalar("val/macro-jaccard", class_report_val["jaccard_macro"], epoch)
        writer.add_scalar("val/micro-jaccard", class_report_val["jaccard_micro"], epoch)

        if class_report_val["sectors"]["macro avg"]["f1-score"] > best_val_macro_f1:
            best_val_macro_f1 = class_report_val["sectors"]["macro avg"]["f1-score"]
            save_model_with_timestamp(classificationhead.state_dict(), PATH)
            best_epoch = epoch

    return best_val_macro_f1, flops * (best_epoch + 1), len(train_dataloader)


def train_evaluate_final(args, flops=False, probas=False):
    _, training_flops, n_train = train_final(
        args.dataset,
        args.lr,
        args.weight_decay,
        args.batch_size,
        1000,
        args.embedding_type,
        args.model_name,
        args.augmented,
        flops,
    )
    PATH = get_path(
        args.dataset,
        args.augmented,
        "classification_head",
        "unconstrained",
        args.model_name,
    )

    test_dataloader, labels, embedding_dim = get_dataloader(
        args.dataset,
        "test",
        args.embedding_type,
        args.batch_size,
        args.model_name,
        args.augmented,
    )
    classificationhead = ClassificationHead(embedding_dim, len(labels))
    classificationhead.load_state_dict(open_most_recent_model(PATH))

    if flops:
        with torch.profiler.profile(with_flops=True) as inference_profiler:
            trues_test, preds_test = validate(
                classificationhead, None, test_dataloader, None, None
            )
        class_report_test = sector_report(trues_test, preds_test, labels)
        class_report_test["training_flops"] = training_flops
        class_report_test["inference_flops"] = sum(
            [event.flops for event in inference_profiler.key_averages()]
        )
    else:
        trues_test, preds_test = validate(
            classificationhead, None, test_dataloader, None, None
        )
        class_report_test = sector_report(trues_test, preds_test, labels)

    if probas:
        trues, probas = get_probs(classificationhead, test_dataloader)
        class_report_test["probas"] = probas
        class_report_test["trues"] = trues
        class_report_test["args"] = vars(args)

    return class_report_test, n_train, len(test_dataloader)
