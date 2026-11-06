"""Document-safe token chunk loading and reproducible preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable, Iterator
import unicodedata

import torch
from torch.utils.data import Dataset


def normalized_text(text: str) -> str:
    """Canonical text for exact duplicate detection, not model tokenization."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def document_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def token_chunk_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(token_ids)}I", *token_ids)).hexdigest()


def hash_partition(doc_hash: str, fractions=(0.8, 0.1, 0.1)) -> str:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("train/validation/test fractions must sum to one")
    u = int(doc_hash[:16], 16) / 2**64
    return "train" if u < fractions[0] else "validation" if u < sum(fractions[:2]) else "test"


@dataclass
class Document:
    text: str
    doc_id: str
    source: str
    source_split: str
    metadata: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return document_hash(self.text)


def reconstruct_wikitext_articles(lines: Iterable[str], source_split: str) -> Iterator[Document]:
    """Join WikiText raw rows at top-level ``= Article title =`` headings.

    Subheadings (``= = Section = =``) remain inside their parent article. Empty
    lines are retained. This preserves source document boundaries before chunking.
    """
    article: list[str] = []
    index = 0
    for line in lines:
        stripped = line.strip()
        title = stripped[1:-1].strip() if stripped.startswith("= ") and stripped.endswith(" =") else ""
        is_heading = bool(title) and not title.startswith("=") and not title.endswith("=")
        if is_heading and article and any(x.strip() for x in article):
            yield Document("\n".join(article), f"wikitext103:{source_split}:{index}",
                           "Salesforce/wikitext:wikitext-103-raw-v1", source_split)
            index += 1
            article = []
        article.append(line)
    if article and any(x.strip() for x in article):
        yield Document("\n".join(article), f"wikitext103:{source_split}:{index}",
                       "Salesforce/wikitext:wikitext-103-raw-v1", source_split)


def chunks_from_token_ids(ids: list[int], sequence_length: int = 512,
                          eos_token_id: int | None = None) -> tuple[list[list[int]], int]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if eos_token_id is not None:
        ids = [*ids, eos_token_id]
    stop = len(ids) // sequence_length * sequence_length
    return [ids[i:i + sequence_length] for i in range(0, stop, sequence_length)], len(ids) - stop


def load_token_dataset(path: str | Path) -> dict:
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "input_ids" not in data:
        raise ValueError("token dataset must be a dictionary containing input_ids")
    ids = data["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("input_ids must be a [N,L] tensor with L >= 2")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have int32 or int64 dtype")
    if "attention_mask" in data and data["attention_mask"].shape != ids.shape:
        raise ValueError("attention_mask shape must match input_ids")
    if "doc_ids" in data and len(data["doc_ids"]) != len(ids):
        raise ValueError("doc_ids must contain one ID per chunk")
    return data


class TokenChunkDataset(Dataset):
    def __init__(self, path_or_data: str | Path | dict):
        self.data = load_token_dataset(path_or_data) if not isinstance(path_or_data, dict) else path_or_data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        item = {"input_ids": self.data["input_ids"][index].long()}
        if "attention_mask" in self.data:
            item["attention_mask"] = self.data["attention_mask"][index]
        return item


def save_token_dataset(path: str | Path, input_ids: list[list[int]] | torch.Tensor,
                       doc_ids: list[str], manifest: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = torch.as_tensor(input_ids, dtype=torch.int32).contiguous()
    if ids.ndim != 2 or ids.shape[1] < 2 or len(doc_ids) != len(ids):
        raise ValueError("invalid token chunks/doc_ids")
    torch.save({"input_ids": ids, "doc_ids": doc_ids}, path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {**manifest, "path": str(path), "sha256": h,
              "chunks": len(ids), "sequence_length": ids.shape[1],
              "raw_input_tokens": ids.numel(),
              "prediction_tokens": len(ids) * (ids.shape[1] - 1),
              "document_count": len(set(doc_ids)), "tensor_dtype": "int32"}
    path.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
