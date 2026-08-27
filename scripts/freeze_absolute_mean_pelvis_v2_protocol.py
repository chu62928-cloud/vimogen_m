#!/usr/bin/env python3
"""Freeze the full-FK, heading-removed sagittal pelvis protocol v2.

The v1 data split is copied byte-for-byte.  No model output is read, so this
amendment cannot change the development/blind boundary after seeing v2 results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT / "results/phase6/absolute_mean_pelvis_v1"
DEFAULT_OUTPUT = ROOT / "results/phase6/absolute_mean_pelvis_v2"
DATA_HASHES = {
    "development20.json": "c45b1fa103cad9b5da2ebae0500ef04e61fb3e95dabcacfbbd06d13cc36fd692",
    "mbench_walking_candidates65.json": "d8cbeb366cf03cfd5e5435eaa7c39ac71f515a05691d068229824ba8524eb603",
    "mbench_primary_blind40.json": "36826ffe83f12995305d5e89fdb6386b750a1c2f3a827228095a13aaf1dafe1c",
    "mbench_robustness450.json": "167f4e131c724abab043121bf2e09f8c5cdebcddfbf497194f8cf44f46d5a17e",
    "formal_video_cases12.json": "e13f6594464d197415e50b4da23966e79ab2d12d63e39a2247961bb451253cd1",
}
SMPLX_NEUTRAL_REST22_RAW_FLOAT32_SHA256 = (
    "4ef73a96e7fb4c06578b84bc6fd915103afa070a82101e1faebb62efe0abb548"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol root: {output}")
    v1_protocol_path = V1_ROOT / "protocol.json"
    if not v1_protocol_path.is_file():
        raise FileNotFoundError(f"missing frozen v1 protocol: {v1_protocol_path}")
    v1_protocol = json.loads(v1_protocol_path.read_text(encoding="utf-8"))
    if v1_protocol.get("protocol") != "vimogen_absolute_mean_pelvis_v1":
        raise RuntimeError("unexpected v1 protocol identity")

    for relative in (
        "data",
        "angles",
        "metrics",
        "summaries",
        "videos/smoke",
        "videos/formal",
        "videos/posthoc",
        "figures",
        "logs",
        "runs",
    ):
        (output / relative).mkdir(parents=True, exist_ok=False)

    copied_hashes: dict[str, str] = {}
    for source in sorted((V1_ROOT / "data").iterdir()):
        if not source.is_file():
            continue
        if source.name in DATA_HASHES:
            actual = sha256(source)
            if actual != DATA_HASHES[source.name]:
                raise RuntimeError(
                    f"frozen v1 split hash mismatch for {source.name}: {actual}"
                )
        destination = output / "data" / source.name
        shutil.copy2(source, destination)
        copied_hashes[source.name] = sha256(destination)

    protocol = {
        "protocol": "vimogen_absolute_mean_pelvis_v2_full_fk_sagittal",
        "status": "FROZEN_BEFORE_V2_MODEL_RUNS",
        "supersedes": {
            "protocol": "vimogen_absolute_mean_pelvis_v1",
            "protocol_path": str(v1_protocol_path),
            "protocol_sha256": sha256(v1_protocol_path),
            "reason": "v1 did not reconstruct J by FK from authoritative pose and used a root-forward pitch proxy without an explicit per-frame heading-removed sagittal frame",
            "v1_results_policy": "retain unchanged as historical engineering evidence; never promote to v2 evidence",
        },
        "target_semantics": "absolute valid-frame mean heading-removed local-sagittal pelvis tilt; +5/+10 are not per-frame constants and not relative deltas",
        "authority_pipeline": [
            "decode authoritative body local rotations",
            "fuse direct root rotation with integrated dR on SO(3)",
            "fuse direct root translation with integrated dT",
            "run differentiable FK from the authoritative pose",
            "replace all 22 direct joint positions J",
            "recompute dJ, dR, and dT by forward differences/relative rotations",
            "pack one internally consistent 276D tensor",
        ],
        "angle_definition": {
            "name": "heading_removed_local_sagittal_pelvis_tilt",
            "local_forward_axis": "+z in the SMPL-X anatomical root frame",
            "world_up_axis": "+z in the ViMoGen canonical frame",
            "steps": [
                "derive the current horizontal character heading from authoritative root orientation",
                "construct the per-frame right, heading, up frame and remove current yaw",
                "project the pelvis anterior direction into the character-local sagittal plane",
                "compute signed tilt with atan2(up component, forward component)",
            ],
            "turn_invariance": "the same local pelvis tilt must be unchanged under any pure yaw/turn sequence",
            "terminal_axis": "G1 left-multiplies each frame around that frame's character-local right axis, then reruns FK and all 276D derivatives",
        },
        "skeleton_authority": {
            "model": "SMPL-X neutral, first 22 body joints, zero betas",
            "parents": [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
            "root_rest_joint": [0.003123254980891943, -0.3514074385166168, 0.012036550790071487],
            "rest22_raw_float32_sha256": SMPLX_NEUTRAL_REST22_RAW_FLOAT32_SHA256,
            "independent_fk_check": "random 4-pose lightweight FK versus smplx 0.1.28: max absolute joint coordinate error 2.3841858e-7, mean joint L2 error 8.9535312e-8",
        },
        "formulas": {
            "clean_endpoint": "x0_hat = x_sigma - sigma * v",
            "fused_root": "R_auth = Exp(MA9(Log(R_direct * transpose(R_velocity)))) * R_velocity",
            "fk_root": "J0 = T_auth + J0_rest",
            "fk_child": "J_i = J_parent + R_global_parent * (J_i_rest - J_parent_rest)",
            "global_child_rotation": "R_global_i = R_global_parent * R_local_i",
            "joint_velocity": "dJ_t = J_(t+1) - J_t",
            "root_rotation_velocity": "dR_t = R_(t+1) * transpose(R_t)",
            "root_translation_velocity": "dT_t = T_(t+1) - T_t",
            "sagittal_tilt": "theta_t = atan2(dot(pelvis_forward_t, up), dot(pelvis_forward_t, heading_t))",
            "mean_angle": "theta_bar = sum(mask * theta) / sum(mask)",
            "loss": "L = (theta_bar - target)^2 + w_shape * mean(mask * ((theta-theta_bar) - (theta_M0-theta_M0_bar))^2) + 0.1 * mean(mask * (x-x_M0)^2)",
            "guided_velocity": "v_guided = (x_sigma - x0_consistent) / sigma",
            "terminal": "G1 is eligible only when abs(target - mean(G0)) <= 1 degree; it applies at most 1 degree around each frame's local right axis and reruns the complete authority-to-276D pipeline",
        },
        "fixed": {
            "targets_deg": [5.0, 10.0],
            "formal_seeds": [0, 1, 2],
            "sigma_window": [0.25, 0.65],
            "max_correction_rms": 0.05,
            "fusion_window": 9,
            "anchor_weight": 1.0,
            "motion_weight": 0.1,
            "terminal_max_deg": 1.0,
            "video": {
                "fps": 20,
                "codec": "h264",
                "resolution": [1920, 1080],
                "camera": "fixed_side",
            },
        },
        "development_grid": {
            "manifest": "data/development20.json",
            "guidance_strength": [0.5, 1.0, 2.0],
            "shape_weight": [0.05, 0.1, 0.2],
            "selection_seed": 0,
            "verification_seeds": [1, 2],
            "blind_data_tuning_forbidden": True,
        },
        "formal": {
            "walking_candidate_count": 65,
            "primary_blind_count": 40,
            "robustness_count": 450,
            "formal_video_case_count": 12,
            "formal_video_count": 24,
            "posthoc_best_worst_labeled": True,
        },
        "success": {
            "per_target_error_median_deg_max": 2.0,
            "per_target_units_le_2deg_rate_min": 0.90,
            "root_rotation_velocity_residual_deg_max": 1e-4,
            "joint_fk_max_abs_m_max": 1e-5,
            "joint_velocity_max_abs_m_max": 1e-6,
            "root_translation_velocity_max_abs_m_max": 1e-6,
            "centered_curve_correlation_median_min": 0.90,
            "fluctuation_std_ratio_range": [0.8, 1.2],
            "mbench_naturalness": "no statistically supported degradation",
        },
        "g1_promotion": {
            "control_more_accurate": True,
            "terminal_trigger_rate_max": 0.05,
            "centered_curve_error_increase_deg_max": 0.1,
            "naturalness_no_degradation": True,
        },
        "failure_review_order": [
            "angle_and_coordinate_convention",
            "authority_boundary",
            "smoothing_attenuation",
            "gradient_and_standardization",
            "loss_conflict",
            "target_feasibility",
            "fusion_window",
            "pelvis_spine_compensation",
        ],
        "data": {
            "selection_reused_byte_for_byte_from_v1": True,
            "files_sha256": copied_hashes,
        },
        "execution_gates": [
            "specialized synthetic tests",
            "single real-motion MP4 smoke",
            "development seed0 grid",
            "development seed1/2 verification",
            "primary blind40 only after development gates",
            "robustness450 only after primary gate",
        ],
    }
    write_json(output / "protocol.json", protocol)
    (output / "README.md").write_text(
        "# ViMoGen absolute mean pelvis v2\n\n"
        "Status: FROZEN_BEFORE_V2_MODEL_RUNS.\n\n"
        "This version reconstructs the complete 276D representation from an "
        "authoritative pose through FK and measures pelvis tilt after removing "
        "the current character heading in the local sagittal plane.\n\n"
        "The v1 directory is retained unchanged and is not v2 evidence. Formal "
        "claims remain pending until all control, consistency, curve, and "
        "naturalness gates pass.\n",
        encoding="utf-8",
    )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = freeze(args.output)
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol": protocol["protocol"],
                "output": str(args.output),
                "protocol_sha256": sha256(args.output / "protocol.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
