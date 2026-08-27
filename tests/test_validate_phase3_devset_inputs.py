import json
from pathlib import Path

import torch

from scripts.validate_phase3_devset_inputs import validate


def test_validates_text_embeddings(tmp_path: Path):
    emb = tmp_path / "1.pt"
    torch.save(torch.zeros(3, 4096, dtype=torch.bfloat16), emb)
    data = tmp_path / "inputs.json"
    data.write_text(
        json.dumps(
            [
                {
                    "sample_id": "1",
                    "global_id": "1",
                    "use_ref_motion": False,
                    "prompt_wanvideot5_embed_path": str(emb),
                }
            ]
            * 20
        ),
        encoding="utf-8",
    )
    # Make IDs unique after constructing the compact fixture.
    rows = json.loads(data.read_text(encoding="utf-8"))
    for index, row in enumerate(rows):
        row["sample_id"] = str(index)
        row["global_id"] = str(index)
    data.write_text(json.dumps(rows), encoding="utf-8")
    audit = validate(data, tmp_path / "audit.json")
    assert audit["status"] == "VERIFIED_TEXT_INPUTS"
    assert len(audit["embeddings"]) == 20
