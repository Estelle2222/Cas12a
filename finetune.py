import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import CasBertConfig, CasBertModel, CasNucleotideTokenizer, CasSequenceDataset
from normalizer import Normalizer

VARIANT_COL = "variant"
ORIGINAL_VARIANT = "original"
REPLACED_VARIANT = "replaced"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataframe_8_1_1(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits.")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val

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
    sequence_col: str,
    label_col: str,
    z_label_col: str,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    df = _read_table(path)

    missing = [c for c in [sequence_col, label_col, z_label_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}"
        )

    df = df[[sequence_col, label_col, z_label_col]].dropna().copy()
    df[sequence_col] = df[sequence_col].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[z_label_col] = pd.to_numeric(df[z_label_col], errors="coerce")
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
    return {
        "train": _augment_split_with_replaced_sequences(
            train_df, sequence_col, label_col, z_label_col
        ),
        "val": _augment_split_with_replaced_sequences(
            val_df, sequence_col, label_col, z_label_col
        ),
        "test": _augment_split_with_replaced_sequences(
            test_df, sequence_col, label_col, z_label_col
        ),
    }


def _augment_split_with_replaced_sequences(
    split_df: pd.DataFrame,
    sequence_col: str,
    label_col: str,
    z_label_col: str,
) -> pd.DataFrame:
    original_df = split_df.copy()
    original_df[VARIANT_COL] = ORIGINAL_VARIANT

    replaced_df = split_df.copy()
    replaced_df[sequence_col] = (
        replaced_df[sequence_col]
        .str.replace("a", "z", regex=False)
        .str.replace("A", "Z", regex=False)
    )
    replaced_df[label_col] = replaced_df[z_label_col]
    replaced_df[VARIANT_COL] = REPLACED_VARIANT

    return pd.concat([original_df, replaced_df], ignore_index=True)


def apply_label_normalization(
    splits: Dict[str, pd.DataFrame],
    label_col: str,
    z_label_col: str,
    normalizer: Normalizer,
) -> Dict[str, pd.DataFrame]:
    normalized_splits: Dict[str, pd.DataFrame] = {}
    for split_name, split_df in splits.items():
        split_copy = split_df.copy()
        split_copy[label_col] = normalizer.normalize(
            split_copy[label_col].astype(float).to_numpy()
        )
        if z_label_col in split_copy.columns:
            split_copy[z_label_col] = normalizer.normalize(
                split_copy[z_label_col].astype(float).to_numpy()
            )
            if VARIANT_COL in split_copy.columns:
                replaced_mask = split_copy[VARIANT_COL] == REPLACED_VARIANT
                split_copy.loc[replaced_mask, label_col] = split_copy.loc[
                    replaced_mask, z_label_col
                ]
        normalized_splits[split_name] = split_copy
    return normalized_splits


def build_dataloaders(
    splits: Dict[str, pd.DataFrame],
    tokenizer: CasNucleotideTokenizer,
    max_len: int,
    batch_size: int,
    sequence_col: str,
    label_col: str,
    num_workers: int,
) -> Dict[str, DataLoader]:
    def _build_loader(split_df: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = CasSequenceDataset(
            sequences=split_df[sequence_col].tolist(),
            y_values=split_df[label_col].astype(float).tolist(),
            tokenizer=tokenizer,
            max_len=max_len,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    dataloaders = {
        "train": _build_loader(splits["train"], shuffle=True),
        "val": _build_loader(splits["val"], shuffle=False),
        "test": _build_loader(splits["test"], shuffle=False),
    }

    for split_name in ["val", "test"]:
        split_df = splits[split_name]
        if VARIANT_COL not in split_df.columns:
            continue

        for variant_name in [ORIGINAL_VARIANT, REPLACED_VARIANT]:
            variant_df = split_df[split_df[VARIANT_COL] == variant_name].reset_index(
                drop=True
            )
            if len(variant_df) == 0:
                continue
            dataloaders[f"{split_name}_{variant_name}"] = _build_loader(
                variant_df,
                shuffle=False,
            )

    return dataloaders


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _spearman_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() < 2 or y_pred.numel() < 2:
        return float("nan")
    true_rank = pd.Series(y_true.detach().cpu().numpy()).rank(method="average")
    pred_rank = pd.Series(y_pred.detach().cpu().numpy()).rank(method="average")
    corr = true_rank.corr(pred_rank, method="pearson")
    return float(corr) if corr is not None else float("nan")


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")


def _config_from_checkpoint(
    checkpoint: Dict,
    tokenizer: CasNucleotideTokenizer,
    max_len_override: int,
    problem_type: str,
    num_outputs: int,
) -> CasBertConfig:
    config_blob = checkpoint.get("config", {})
    if not isinstance(config_blob, dict):
        config_blob = {}

    valid_keys = set(CasBertConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in config_blob.items() if k in valid_keys}

    # Force values needed for this finetuning task.
    cfg_kwargs["vocab_size"] = tokenizer.vocab_size
    cfg_kwargs["pad_token_id"] = tokenizer.pad_id
    cfg_kwargs["problem_type"] = problem_type
    cfg_kwargs["num_outputs"] = num_outputs

    if max_len_override > 0:
        cfg_kwargs["max_len"] = max_len_override
    elif "max_len" not in cfg_kwargs:
        cfg_kwargs["max_len"] = 36

    for k, default in {
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 8,
        "ff_dim": 512,
        "dropout": 0.1,
    }.items():
        cfg_kwargs.setdefault(k, default)

    return CasBertConfig(**cfg_kwargs)


def freeze_all_but_prediction_head(model: CasBertModel) -> Tuple[int, int]:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.prediction_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def copy_token_embedding(model: CasBertModel, source_token_id: int, target_token_id: int) -> None:
    with torch.no_grad():
        model.token_embeddings.weight[target_token_id].copy_(
            model.token_embeddings.weight[source_token_id]
        )


def enable_single_token_embedding_training(
    model: CasBertModel, token_id: int
) -> Tuple[torch.utils.hooks.RemovableHandle, int]:
    emb_weight = model.token_embeddings.weight
    emb_weight.requires_grad = True

    grad_mask = torch.zeros_like(emb_weight)
    grad_mask[token_id] = 1.0

    hook = emb_weight.register_hook(lambda grad: grad * grad_mask)
    trainable_dims = int(emb_weight.shape[1])
    return hook, trainable_dims


def set_head_only_train_mode(model: CasBertModel) -> None:
    # Keep frozen backbone in eval mode (no dropout updates), head in train mode.
    model.eval()
    model.prediction_head.train()


def train_one_epoch_head_only(
    model: CasBertModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    set_head_only_train_mode(model)
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
            clip_params = [p for p in model.parameters() if p.requires_grad]
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
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


def _format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])


def _is_better(current: Dict[str, float], best: float, problem_type: str) -> bool:
    if problem_type == "regression":
        return current["loss"] < best
    return current.get("accuracy", float("-inf")) > best


def resolve_normalizer(
    pretrained_path: Path,
    normalizer_path: Optional[Path],
) -> Tuple[Normalizer, Path]:
    source = normalizer_path if normalizer_path is not None else pretrained_path.parent / "normalizer.json"
    if not source.exists():
        raise FileNotFoundError(
            "Training normalizer json not found. Expected at "
            f"{source}. Pass --normalizer to the exact normalizer.json from training."
        )
    return Normalizer.load_json(source), source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Head-only finetuning for CasBertModel: freeze backbone and train only "
            "prediction_head."
        )
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("outputs/casbert_experiment/best_model.pt"),
        help="Path to pretrained checkpoint.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/Z_gRNA.csv"))
    parser.add_argument("--sequence-col", type=str, default="sequence")
    parser.add_argument("--label-col", type=str, default="efficiency")
    parser.add_argument("--z-label-col", type=str, default="Z-avg")
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=None,
        help=(
            "Path to training normalizer json. If omitted, uses "
            "<pretrained_dir>/normalizer.json."
        ),
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--problem-type",
        type=str,
        default="regression",
        choices=["regression", "binary", "multiclass"],
    )
    parser.add_argument("--num-outputs", type=int, default=1)
    parser.add_argument(
        "--max-len",
        type=int,
        default=-1,
        help="Override max_len from checkpoint config. Use -1 to keep checkpoint value.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="casbert_z_grna_head_only",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if args.output_dir is None:
        args.output_dir = args.pretrained.parent
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CasNucleotideTokenizer()
    checkpoint = _load_checkpoint(args.pretrained, device)
    config = _config_from_checkpoint(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        max_len_override=args.max_len,
        problem_type=args.problem_type,
        num_outputs=args.num_outputs,
    )
    model = CasBertModel(config).to(device)

    load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_msg.missing_keys:
        print(f"Missing keys while loading pretrained model: {load_msg.missing_keys}")
    if load_msg.unexpected_keys:
        print(f"Unexpected keys while loading pretrained model: {load_msg.unexpected_keys}")

    head_trainable_params, total_params = freeze_all_but_prediction_head(model)
    a_token_id = tokenizer.vocab["a"]
    z_token_id = tokenizer.vocab["z"]
    copy_token_embedding(model, source_token_id=a_token_id, target_token_id=z_token_id)
    z_embedding_hook, z_trainable_dims = enable_single_token_embedding_training(
        model, token_id=z_token_id
    )
    trainable_params = head_trainable_params + z_trainable_dims

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.prediction_head.parameters()),
                "weight_decay": args.weight_decay,
            },
            {
                # Keep weight decay off for embedding matrix since only one row is trainable.
                "params": [model.token_embeddings.weight],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )
    _ = z_embedding_hook

    splits_raw = load_and_prepare_data(
        path=args.data,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        z_label_col=args.z_label_col,
        seed=args.seed,
    )

    normalizer: Optional[Normalizer] = None
    splits_for_training = splits_raw
    if args.problem_type == "regression":
        normalizer, source_normalizer_path = resolve_normalizer(
            pretrained_path=args.pretrained,
            normalizer_path=args.normalizer,
        )

        normalizer_path = run_dir / "normalizer.json"
        normalizer.save_json(normalizer_path)
        splits_for_training = apply_label_normalization(
            splits=splits_raw,
            label_col=args.label_col,
            z_label_col=args.z_label_col,
            normalizer=normalizer,
        )
        print(
            f"Using training normalizer: {source_normalizer_path} "
            f"(mean={normalizer.mean:.6f}, std={normalizer.std:.6f})"
        )
        print(f"Copied normalizer to: {normalizer_path}")

    dataloaders = build_dataloaders(
        splits=splits_for_training,
        tokenizer=tokenizer,
        max_len=config.max_len,
        batch_size=args.batch_size,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        num_workers=args.num_workers,
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

    print(f"Pretrained checkpoint: {args.pretrained}")
    print(f"Finetune data: {args.data}")
    train_original_size = int(
        (splits_raw["train"][VARIANT_COL] == ORIGINAL_VARIANT).sum()
    )
    val_original_size = int((splits_raw["val"][VARIANT_COL] == ORIGINAL_VARIANT).sum())
    test_original_size = int(
        (splits_raw["test"][VARIANT_COL] == ORIGINAL_VARIANT).sum()
    )
    print(
        "Split sizes after doubling (train/val/test): "
        f"{len(splits_raw['train'])}/{len(splits_raw['val'])}/{len(splits_raw['test'])}"
    )
    print(
        "Original counts before doubling (train/val/test): "
        f"{train_original_size}/{val_original_size}/{test_original_size}"
    )
    print(f"Device: {device}")
    print(f"Trainable parameters (head + z-token embedding): {trainable_params:,}/{total_params:,}")
    print(f"Initialized embedding 'z' from 'a' (token ids: a={a_token_id}, z={z_token_id})")

    best_metric = float("inf") if args.problem_type == "regression" else float("-inf")
    best_model_path = run_dir / "best_model.pt"
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop_patience = 20

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_head_only(
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
        val_original_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val_original"],
            device=device,
            problem_type=args.problem_type,
            normalizer=normalizer,
        )
        val_replaced_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val_replaced"],
            device=device,
            problem_type=args.problem_type,
            normalizer=normalizer,
        )

        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}
        for k, v in val_metrics.items():
            row[f"val_{k}"] = float(v)
        for k, v in val_original_metrics.items():
            row[f"val_original_{k}"] = float(v)
        for k, v in val_replaced_metrics.items():
            row[f"val_replaced_{k}"] = float(v)
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val(all): {_format_metrics(val_metrics)} | "
            f"val(original): {_format_metrics(val_original_metrics)} | "
            f"val(replaced): {_format_metrics(val_replaced_metrics)}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

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
                    "source_pretrained": str(args.pretrained),
                    "trainable_part": "prediction_head_only",
                    "normalizer": normalizer.to_dict() if normalizer is not None else None,
                },
                best_model_path,
            )
            print(f"  Saved new best model to {best_model_path}")

        if no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: validation loss did not improve for "
                f"{early_stop_patience} consecutive epochs."
            )
            break

    history_path = run_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved finetune history: {history_path}")

    if not best_model_path.exists():
        raise RuntimeError("Best model checkpoint was not created.")

    best_ckpt = _load_checkpoint(best_model_path, device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        problem_type=args.problem_type,
        normalizer=normalizer,
    )
    test_original_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test_original"],
        device=device,
        problem_type=args.problem_type,
        normalizer=normalizer,
    )
    test_replaced_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test_replaced"],
        device=device,
        problem_type=args.problem_type,
        normalizer=normalizer,
    )
    print(f"Best validation metric: {best_metric:.6f}")
    print(f"Best model epoch: {best_metric_epoch}")
    best_metric_path = run_dir / "best_metric.txt"
    print(f"Saved best validation metric to: {best_metric_path}")
    print(f"Test metrics (all): {_format_metrics(test_metrics)}")
    print(f"Test metrics (original): {_format_metrics(test_original_metrics)}")
    print(f"Test metrics (replaced): {_format_metrics(test_replaced_metrics)}")
    with best_metric_path.open("w") as f:
        f.write(f"{best_metric:.6f}")
        f.write(f"\nBest model epoch: {best_metric_epoch}\n")
        f.write(f"Test metrics (all): {_format_metrics(test_metrics)}\n")
        f.write(f"Test metrics (original): {_format_metrics(test_original_metrics)}\n")
        f.write(f"Test metrics (replaced): {_format_metrics(test_replaced_metrics)}\n")


if __name__ == "__main__":
    main()
