import os
import glob
import numpy as np
import pandas as pd
import json


def calculate_and_finalize(
    grouped_data: pd.DataFrame, agg_rules: dict, weighted_avg_cols: list
) -> pd.DataFrame:
    agg_result = grouped_data.agg(agg_rules)
    for col in weighted_avg_cols:
        weighted_sum_col = f"{col}_weighted"
        agg_result[col] = agg_result[weighted_sum_col] / agg_result["eval_num"]
        agg_result = agg_result.drop(columns=[weighted_sum_col])
    return agg_result.reset_index()


def analyze_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # remove any column whose name contains "id"
    cols_to_drop = [col for col in data.columns if "id" in col]
    data = data.drop(columns=cols_to_drop)

    sum_cols = ["eval_num"]
    weighted_avg_cols = []
    for col in data.columns:
        if (
            "_rate" in col
            or "_avg" in col
            or "_length" in col
            or "prm_road_map_nodes" in col
        ):
            weighted_avg_cols.append(col)

    numeric_cols = sum_cols + weighted_avg_cols
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in weighted_avg_cols:
        data[f"{col}_weighted"] = data[col] * data["eval_num"]

    agg_rules = {}
    for col in sum_cols:
        agg_rules[col] = "sum"
    for col in weighted_avg_cols:
        agg_rules[f"{col}_weighted"] = "sum"

    result = calculate_and_finalize(
        data.groupby(["scene_name", "num_sections"]), agg_rules, weighted_avg_cols
    )
    result_scene = calculate_and_finalize(
        data.groupby("scene_name"), agg_rules, weighted_avg_cols
    )
    result_section = calculate_and_finalize(
        data.groupby("num_sections"), agg_rules, weighted_avg_cols
    )

    return result, result_scene, result_section


def save_log_to_csv(dir_path: str):
    search_path = os.path.join(dir_path, "*", "*_results.json")
    result_files = glob.glob(search_path)
    records = []
    for json_file in sorted(result_files):
        with open(json_file, "r") as fh:
            data = json.load(fh)
            for k, v in list(data.items()):
                data[k] = json.dumps(v, ensure_ascii=False)
            records.append(data)

    combined_df = pd.DataFrame(records)
    result, result_scene, result_section = analyze_data(combined_df)

    result.to_csv(f"{dir_path}/result.csv", index=False)
    result_scene.to_csv(f"{dir_path}/result_scene.csv", index=False)
    result_section.to_csv(f"{dir_path}/result_section.csv", index=False)

    print(f"Aggregated {len(records)} JSON files into {dir_path}")


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
    dir_paths = [
        "results/mp_eval_max_iter_3_cpu"
        # "results/mp_test_filtered",
        # "results/mp_test_cpu",
        # "results/mp_test_iter_1",
        # "results/mp_test_iter_5",
        # "results/mp_test_iter_20",
        # "results/mp_test_ori",
    ]
    for dir_path in dir_paths:
        save_log_to_csv(dir_path)
