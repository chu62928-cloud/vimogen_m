#!/usr/bin/env python3
"""Validate encoded pure-text development-set inputs without running ViMoGen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(input_path: Path, audit_path: Path) -> dict[str, Any]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if len(rows) != 20:
        raise ValueError(f"expected 20 rows, found {len(rows)}")
    ids = [str(row.get("sample_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    embedding_records = []
    for row in rows:
        if row.get("use_ref_motion") is not False:
            raise ValueError(f"row {row.get('global_id')} is not explicitly text-only")
        path = Path(row["prompt_wanvideot5_embed_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        embedding = torch.load(path, weights_only=True, map_location="cpu")
        if embedding.ndim != 2 or embedding.shape[1] != 4096 or not (0 < embedding.shape[0] <= 226):
            raise ValueError(f"invalid embedding shape for {row.get('global_id')}: {tuple(embedding.shape)}")
        embedding_records.append(
            {
                "sample_id": str(row["sample_id"]),
                "path": str(path),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype),
                "sha256": _sha256(path),
            }
        )
    audit = {
        "schema_version": "phase3_devset_input_audit_v1",
        "status": "VERIFIED_TEXT_INPUTS",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "target_size": len(rows),
        "condition": "text_only",
        "reference_motion_used": False,
        "embeddings": embedding_records,
    }
    if audit_path.exists():
        raise FileExistsError(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.input, args.audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
