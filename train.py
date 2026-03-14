import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import (
    CasBertConfig,
    CasBertModel,
    CasNucleotideTokenizer,
    CasSequenceDataset,
)
from normalizer import Normalizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataframe_8_1_1(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle and split into train/val/test with ratio 8:1:1.
    """
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits.")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    # Guarantee non-empty val/test for small datasets.
    if n_val == 0:
        n_val = 1
        n_train -= 1
        n_test = n_total - n_train - n_val
    if n_test == 0:
        n_test = 1
        n_train -= 1

    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}. Use .csv or .xlsx/.xls")


def load_and_prepare_data(
    path: Path,
    sequence_col: str = "sequence",
    label_col: str = "Indel",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    df = _read_table(path)

    missing = [c for c in [sequence_col, label_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}"
        )

    df = df[[sequence_col, label_col]].dropna().copy()
    df[sequence_col] = df[sequence_col].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    allowed_chars = set("ATGCZatgcz")
    valid_mask = df[sequence_col].apply(
        lambda s: len(s) > 0 and set(s).issubset(allowed_chars)
    )
    df = df[valid_mask].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(
            "Not enough valid rows after cleaning. Need at least 3 rows for 8:1:1 split."
        )

    train_df, val_df, test_df = split_dataframe_8_1_1(df, seed=seed)
    return {"train": train_df, "val": val_df, "test": test_df}


def apply_label_normalization(
    splits: Dict[str, pd.DataFrame],
    label_col: str,
    normalizer: Normalizer,
) -> Dict[str, pd.DataFrame]:
    normalized_splits: Dict[str, pd.DataFrame] = {}
    for split_name, split_df in splits.items():
        split_copy = split_df.copy()
        split_copy[label_col] = normalizer.normalize(
            split_copy[label_col].astype(float).to_numpy()
        )
        normalized_splits[split_name] = split_copy
    return normalized_splits


def build_dataloaders(
    splits: Dict[str, pd.DataFrame],
    tokenizer: CasNucleotideTokenizer,
    max_len: int,
    batch_size: int,
    sequence_col: str = "sequence",
    label_col: str = "Indel",
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    datasets = {}
    for split_name, split_df in splits.items():
        datasets[split_name] = CasSequenceDataset(
            sequences=split_df[sequence_col].tolist(),
            y_values=split_df[label_col].astype(float).tolist(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

    use_pin_memory = torch.cuda.is_available()
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
    }


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _spearman_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() < 2 or y_pred.numel() < 2:
        return float("nan")
    true_rank = pd.Series(y_true.detach().cpu().numpy()).rank(method="average")
    pred_rank = pd.Series(y_pred.detach().cpu().numpy()).rank(method="average")
    corr = true_rank.corr(pred_rank, method="pearson")
    return float(corr) if corr is not None else float("nan")


def train_one_epoch(
    model: CasBertModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output["loss"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = batch["labels"].shape[0]
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: CasBertModel,
    dataloader: DataLoader,
    device: torch.device,
    problem_type: str,
    normalizer: Optional[Normalizer] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output["loss"]

        bs = batch["labels"].shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(output["logits"].detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())

    metrics: Dict[str, float] = {"loss": total_loss / max(total_samples, 1)}
    if not all_logits:
        return metrics

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if problem_type == "regression":
        preds_norm = logits.squeeze(-1) if logits.ndim > 1 and logits.shape[-1] == 1 else logits
        target_norm = labels.float().view_as(preds_norm)

        if normalizer is not None:
            preds = normalizer.denormalize(preds_norm)
            target = normalizer.denormalize(target_norm)
        else:
            preds = preds_norm
            target = target_norm

        mse = torch.mean((preds - target) ** 2).item()
        mae = torch.mean(torch.abs(preds - target)).item()
        ss_res = torch.sum((preds - target) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        spearman = _spearman_corr(target, preds)
        metrics.update({"mse": mse, "mae": mae, "r2": r2.item(), "spearman": spearman})
        return metrics

    if problem_type == "binary":
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs >= 0.5).float()
        target = labels.float().view(-1)
        accuracy = (preds == target).float().mean().item()
        metrics.update({"accuracy": accuracy})
        return metrics

    if problem_type == "multiclass":
        preds = torch.argmax(logits, dim=-1).view(-1)
        target = labels.long().view(-1)
        accuracy = (preds == target).float().mean().item()
        metrics.update({"accuracy": accuracy})
        return metrics

    raise ValueError(
        f"Unsupported problem_type: {problem_type}. "
        "Use 'regression', 'binary', or 'multiclass'."
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CasBertModel using an 8:1:1 train/val/test split."
    )
    parser.add_argument("--run-name", type=str, default="casbert_experiment", help="Name for this training run.")
    parser.add_argument("--data", type=Path, default=Path("data/34bp.csv"))
    parser.add_argument("--sequence-col", type=str, default="sequence")
    parser.add_argument("--label-col", type=str, default="Indel")
    parser.add_argument("--max-len", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["regression", "binary", "multiclass"],
        default="regression",
    )
    parser.add_argument("--num-outputs", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/"),
        help="Directory to save splits, checkpoints, and logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def _format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.6f}")
    return ", ".join(parts)


def _is_better(current: Dict[str, float], best: float, problem_type: str) -> bool:
    if problem_type == "regression":
        # Lower val loss is better.
        return current["loss"] < best
    # Higher val accuracy is better.
    return current.get("accuracy", float("-inf")) > best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits_raw = load_and_prepare_data(
        path=args.data,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        seed=args.seed,
    )

    normalizer: Optional[Normalizer] = None
    splits_for_training = splits_raw
    if args.problem_type == "regression":
        normalizer = Normalizer().fit(
            splits_raw["train"][args.label_col].astype(float).to_numpy()
        )
        normalizer_path = run_dir / "normalizer.json"
        normalizer.save_json(normalizer_path)
        splits_for_training = apply_label_normalization(
            splits=splits_raw,
            label_col=args.label_col,
            normalizer=normalizer,
        )
        print(
            f"Saved normalizer to {normalizer_path} "
            f"(mean={normalizer.mean:.6f}, std={normalizer.std:.6f})"
        )

    tokenizer = CasNucleotideTokenizer()
    dataloaders = build_dataloaders(
        splits=splits_for_training,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        num_workers=args.num_workers,
    )

    config = CasBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        num_outputs=args.num_outputs,
        problem_type=args.problem_type,
        pad_token_id=tokenizer.pad_id,
    )
    model = CasBertModel(config).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    config_path = run_dir / "model_config.json"
    with config_path.open("w") as f:
        f.write(json.dumps(vars(config), indent=4))
    print(f"Saved model config to {config_path}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_raw["train"].to_csv(splits_dir / "train.csv", index=False)
    splits_raw["val"].to_csv(splits_dir / "val.csv", index=False)
    splits_raw["test"].to_csv(splits_dir / "test.csv", index=False)

    if normalizer is not None:
        norm_splits_dir = run_dir / "splits_normalized"
        norm_splits_dir.mkdir(parents=True, exist_ok=True)
        splits_for_training["train"].to_csv(norm_splits_dir / "train.csv", index=False)
        splits_for_training["val"].to_csv(norm_splits_dir / "val.csv", index=False)
        splits_for_training["test"].to_csv(norm_splits_dir / "test.csv", index=False)

    print(f"Data file: {args.data}")
    print(
        f"Split sizes (train/val/test): "
        f"{len(splits_raw['train'])}/{len(splits_raw['val'])}/{len(splits_raw['test'])}"
    )
    print(f"Device: {device}")

    best_metric = float("inf") if args.problem_type == "regression" else float("-inf")
    best_model_path = run_dir / "best_model.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            problem_type=args.problem_type,
            normalizer=normalizer,
        )

        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}
        for k, v in val_metrics.items():
            row[f"val_{k}"] = float(v)
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val: {_format_metrics(val_metrics)}"
        )

        if _is_better(val_metrics, best_metric, args.problem_type):
            best_metric = (
                val_metrics["loss"]
                if args.problem_type == "regression"
                else val_metrics["accuracy"]
            )
            best_metric_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": vars(config),
                    "args": vars(args),
                    "normalizer": normalizer.to_dict() if normalizer is not None else None,
                },
                best_model_path,
            )
            print(f"  Saved new best model to {best_model_path}")

    history_path = run_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    if not best_model_path.exists():
        raise RuntimeError("Best model checkpoint was not created.")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        problem_type=args.problem_type,
        normalizer=normalizer,
    )
    print(f"Best validation metric: {best_metric:.6f}")
    print(f"Best model found at epoch {best_metric_epoch}")
    print(f"Test metrics: {_format_metrics(test_metrics)}")
    best_metrics_path = run_dir / "best_model_metrics.txt"
    with open(best_metrics_path, "w") as f:
        f.write(f"Best validation metric: {best_metric:.6f}\n")
        f.write(f"Best model found at epoch {best_metric_epoch}\n")
        f.write(f"Test metrics: {_format_metrics(test_metrics)}\n")
    print(f"Saved best model metrics: {best_metrics_path}")


if __name__ == "__main__":
    main()
