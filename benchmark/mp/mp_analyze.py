import os

import numpy as np
import pandas as pd


def error_calculate(reference_csv_path: str, planned_csv_path: str) -> dict:
    """
    Compute per-timestep position and rotation errors from reference and planned CSVs.

    The CSVs must contain columns: x, y, z, qw, qx, qy, qz (quaternion in wxyz order).
    Errors are aligned by the row index; if the files have different lengths, the
    shorter length is used.

    Returns a dictionary with arrays and summary statistics:
      - position_errors_mm, rotation_errors_deg (converted arrays)
      - position_error_mean_mm, position_error_std_mm
      - rotation_error_mean_deg, rotation_error_std_deg
    """
    required_columns = ["x", "y", "z", "qw", "qx", "qy", "qz"]

    def _load_and_validate(csv_path: str) -> "pd.DataFrame":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {', '.join(missing)}"
            )
        return df

    def _quat_wxyz_to_rotmat(q: "np.ndarray") -> "np.ndarray":
        # Follows the same convention as benchmark/mp_plot.py
        w, x, y, z = q
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm > 0:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    ref_df = _load_and_validate(reference_csv_path)
    plan_df = _load_and_validate(planned_csv_path)

    # Align length
    T = min(len(ref_df), len(plan_df))
    if T == 0:
        raise ValueError("Empty CSV(s): no rows to compare")

    ref_df = ref_df.iloc[:T].reset_index(drop=True)
    plan_df = plan_df.iloc[:T].reset_index(drop=True)

    # Extract positions and quaternions
    ref_positions = ref_df[["x", "y", "z"]].to_numpy(dtype=float)
    plan_positions = plan_df[["x", "y", "z"]].to_numpy(dtype=float)

    ref_quats_wxyz = ref_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    plan_quats_wxyz = plan_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    # Position errors (meters)
    position_errors_m = np.linalg.norm(plan_positions - ref_positions, axis=1)

    # Rotation errors (radians) via R_rel = R_ref^T * R_plan, angle = acos((trace(R_rel) - 1)/2)
    ref_rot_mats = np.stack([_quat_wxyz_to_rotmat(q) for q in ref_quats_wxyz], axis=0)
    plan_rot_mats = np.stack([_quat_wxyz_to_rotmat(q) for q in plan_quats_wxyz], axis=0)
    R_rel = np.einsum(
        "tij,tjk->tik", np.transpose(ref_rot_mats, (0, 2, 1)), plan_rot_mats
    )
    traces = np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_errors_rad = np.arccos(traces)

    # Summary statistics (SI units)
    position_error_mean_m = float(np.nanmean(position_errors_m))
    position_error_std_m = float(np.nanstd(position_errors_m))
    rotation_error_mean_rad = float(np.nanmean(rotation_errors_rad))
    rotation_error_std_rad = float(np.nanstd(rotation_errors_rad))

    # Convenience conversions
    position_errors_mm = position_errors_m * 1000.0
    rotation_errors_deg = np.rad2deg(rotation_errors_rad)
    position_error_mean_mm = float(np.nanmean(position_errors_mm))
    position_error_std_mm = float(np.nanstd(position_errors_mm))
    rotation_error_mean_deg = float(np.nanmean(rotation_errors_deg))
    rotation_error_std_deg = float(np.nanstd(rotation_errors_deg))

    # Concise printout
    print("--- Error Summary (aligned by row index) ---")
    print(f"Timesteps compared: {T}")
    print(
        f"Position Error (mean ± std): {position_error_mean_mm:.4f} ± {position_error_std_mm:.4f} mm"
    )
    print(
        f"Rotation Error (mean ± std): {rotation_error_mean_deg:.4f} ± {rotation_error_std_deg:.4f} deg"
    )

    return {
        "position_errors_mm": position_errors_mm,
        "position_error_mean_mm": position_error_mean_mm,
        "position_error_std_mm": position_error_std_mm,
        "rotation_errors_deg": rotation_errors_deg,
        "rotation_error_mean_deg": rotation_error_mean_deg,
        "rotation_error_std_deg": rotation_error_std_deg,
    }


if __name__ == "__main__":
    pass
