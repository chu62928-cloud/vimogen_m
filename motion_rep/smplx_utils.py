"""Small, dependency-free helpers for locating the local SMPL-X assets.

The v3 tools are intentionally usable from a clean worktree.  Older analysis
scripts import a large ``motion_checker`` module that is only present in the
interactive experiment workspace, so the new tools use this narrow helper
instead.
"""

from __future__ import annotations

import os
from pathlib import Path


def default_smpl_model_path(kind: str = "smplx", root: str | Path | None = None) -> str:
    """Return the repository-local body-model directory.

    ``VIMOGEN_ROOT`` is honoured for server jobs whose code and read-only data
    live in different locations.  The path is returned even before it exists;
    the SMPL-X constructor then emits the useful missing-asset error.
    """

    roots: list[Path] = []
    if root is not None:
        roots.append(Path(root))
    env_root = os.environ.get("VIMOGEN_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
    seen: set[Path] = set()
    for base in roots:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        candidate = base / "data" / "body_models" / kind
        if candidate.exists():
            return str(candidate)
    return str(roots[0] / "data" / "body_models" / kind)


__all__ = ["default_smpl_model_path"]
