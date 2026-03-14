import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np
import torch


ArrayLike = Union[Iterable[float], np.ndarray, torch.Tensor]


class Normalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-12) -> None:
        self.mean = float(mean)
        self.std = float(std) if float(std) > eps else 1.0
        self.eps = float(eps)

    def fit(self, values: ArrayLike) -> "Normalizer":
        arr = self._to_numpy(values)
        if arr.size == 0:
            raise ValueError("Cannot fit Normalizer on empty values.")
        self.mean = float(arr.mean())
        std = float(arr.std())
        self.std = std if std > self.eps else 1.0
        return self

    def normalize(self, values: ArrayLike) -> ArrayLike:
        if isinstance(values, torch.Tensor):
            return (values - self.mean) / self.std
        arr = self._to_numpy(values)
        return (arr - self.mean) / self.std

    def denormalize(self, values: ArrayLike) -> ArrayLike:
        if isinstance(values, torch.Tensor):
            return values * self.std + self.mean
        arr = self._to_numpy(values)
        return arr * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Normalizer":
        if "mean" not in payload or "std" not in payload:
            raise KeyError("Normalizer payload must contain 'mean' and 'std'.")
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "Normalizer":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    @staticmethod
    def _to_numpy(values: ArrayLike) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values.astype(np.float32, copy=False)
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(list(values), dtype=np.float32)
