import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from torch.utils.data import Dataset


class CasNucleotideTokenizer:
    """
    Tokenizer for [CLS] sequence [SEP] style inputs.
    Supports both lowercase and uppercase nucleotide letters.
    """

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    NUCLEOTIDE_TOKENS = ["a", "t", "g", "c", "z"]

    def __init__(self) -> None:
        canonical_tokens = self.SPECIAL_TOKENS + self.NUCLEOTIDE_TOKENS
        self.vocab: Dict[str, int] = {
            tok: idx for idx, tok in enumerate(canonical_tokens)
        }

        # Map uppercase nucleotide symbols to the same ids.
        for tok in self.NUCLEOTIDE_TOKENS:
            self.vocab[tok.upper()] = self.vocab[tok]

        self.id_to_token: Dict[int, str] = {
            idx: tok for idx, tok in enumerate(canonical_tokens)
        }

        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

    @property
    def vocab_size(self) -> int:
        # Return canonical vocab size (without uppercase aliases).
        return len(self.SPECIAL_TOKENS) + len(self.NUCLEOTIDE_TOKENS)

    def encode(self, sequence: str, max_len: int) -> Dict[str, List[int]]:
        if max_len < 4:
            raise ValueError("max_len must be >= 4 to fit [CLS], token, token, [SEP]")

        seq_tokens = list(str(sequence))
        tokens = ["[CLS]"] + seq_tokens + ["[SEP]"]

        input_ids = [self.vocab.get(t, self.vocab.get(t.lower(), self.unk_id)) for t in tokens]

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            input_ids[-1] = self.sep_id

        attention_mask = [1] * len(input_ids)

        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class CasSequenceDataset(Dataset):
    """
    Expects aligned (sequence, y) entries.
    """

    def __init__(
        self,
        sequences: Sequence[str],
        y_values: Sequence[float],
        tokenizer: CasNucleotideTokenizer,
        max_len: int,
    ) -> None:
        if not (len(sequences) == len(y_values)):
            raise ValueError("sequences and y_values must have the same length")

        self.sequences = list(sequences)
        self.y_values = list(y_values)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.y_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer.encode(
            sequence=self.sequences[idx],
            max_len=self.max_len,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.y_values[idx], dtype=torch.float32),
        }


@dataclass
class CasBertConfig:
    vocab_size: int
    max_len: int = 64
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1
    num_outputs: int = 1
    problem_type: str = "regression"  # "regression", "binary", "multiclass"
    pad_token_id: int = 0


class CasBertModel(nn.Module):
    """
    BERT-like encoder with a pooled-token prediction head.
    """

    def __init__(self, config: CasBertConfig) -> None:
        super().__init__()
        self.config = config
        self.problem_type = config.problem_type

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_len, config.hidden_size)
        self.embed_norm = nn.LayerNorm(config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_outputs),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pooled_token_index: int = 0,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds config.max_len ({self.config.max_len})"
            )

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if pooled_token_index >= seq_len:
            raise ValueError(
                f"pooled_token_index ({pooled_token_index}) must be < seq_len ({seq_len})"
            )
        pooled_hidden = x[:, pooled_token_index, :]
        logits = self.prediction_head(pooled_hidden)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = self._compute_loss(logits, labels)
        return output

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.problem_type == "regression":
            if logits.shape[-1] == 1:
                return F.mse_loss(logits.squeeze(-1), labels.float().view(-1))
            return F.mse_loss(logits, labels.float())

        if self.problem_type == "binary":
            target = labels.float().view(-1, 1)
            return F.binary_cross_entropy_with_logits(logits, target)

        if self.problem_type == "multiclass":
            return F.cross_entropy(logits, labels.long().view(-1))

        raise ValueError(
            f"Unsupported problem_type: {self.problem_type}. "
            "Use 'regression', 'binary', or 'multiclass'."
        )


if __name__ == "__main__":
    # Smoke test.
    tokenizer = CasNucleotideTokenizer()

    batch_seqs = ["aTGczatgc", "GGGtttAAA"]
    batch_y = [0.42, 0.75]

    dataset = CasSequenceDataset(
        sequences=batch_seqs,
        y_values=batch_y,
        tokenizer=tokenizer,
        max_len=36,
    )

    batch = [dataset[0], dataset[1]]
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)

    cfg = CasBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=36,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
        num_outputs=1,
        problem_type="regression",
        pad_token_id=tokenizer.pad_id,
    )
    model = CasBertModel(cfg)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("logits shape:", out["logits"].shape)
    print("loss:", float(out["loss"].item()))
