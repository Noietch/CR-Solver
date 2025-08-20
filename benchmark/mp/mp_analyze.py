import argparse
import glob
import os

import numpy as np
import pandas as pd


def _calculate_group_summary(group: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """
    Helper function to compute summary statistics for a given data group (DataFrame).
    In this version, position and rotation errors are calculated only for successful cases.
    """
    actual_eval_num = len(group)
    if actual_eval_num == 0:
        return pd.Series()

    # --- Accuracy calculation (using all cases) ---
    successful_trials = group["Success Rate (%)"] == 100
    pos_ok = group["Position Error (m)"] < 0.08
    rot_ok = group["Rotation Error (rad)"] < 0.9
    accurate_trials_count = np.sum(successful_trials & pos_ok & rot_ok)
    accuracy = (accurate_trials_count / actual_eval_num) * 100

    # --- Filter successful cases for error analysis ---
    # Use group["Success Rate (%)"] == 100 to filter
    successful_group = group[group["Success Rate (%)"] == 100]

    summary = {
        "Eval Num": actual_eval_num,
        "Accuracy (%)": accuracy,
        "Success Rate Mean (%)": group["Success Rate (%)"].mean(),
        "Success Rate Std (%)": group["Success Rate (%)"].std(),
        "Kinematic Reachability Rate Mean (%)": group[
            "Kinematic Reachability Rate (%)"
        ].mean(),
        "Kinematic Reachability Rate Std (%)": group[
            "Kinematic Reachability Rate (%)"
        ].std(),
        # If there are no successful cases, .mean()/.std() will automatically return NaN, which is correct
        "Position Error Mean (m)": successful_group["Position Error (m)"].mean(),
        "Position Error Std (m)": successful_group["Position Error (m)"].std(),
        "Rotation Error Mean (rad)": successful_group["Rotation Error (rad)"].mean(),
        "Rotation Error Std (rad)": successful_group["Rotation Error (rad)"].std(),
        # Time is usually calculated for all cases to reflect average duration
        "Time Mean (s)": group["Time (s)"].mean(),
        "Time Std (s)": group["Time (s)"].std(),
    }

    # Dynamically handle failure statistics columns
    failure_cols = [
        col for col in group.columns if "Rate" in col and col not in summary
    ]
    for col in failure_cols:
        base_col_name = col.replace(" (%)", "")
        if base_col_name in group.columns:
            mean_val = group[base_col_name].mean()
            summary_col_name = base_col_name.replace("Rate", "Rate Mean") + " (%)"
            summary[summary_col_name] = mean_val

    return pd.Series(summary)


def save_results(results_dir: str, detailed_csv_path: str):
    search_path = os.path.join(
        results_dir, "all_trials_results", "*_all_trials_results.npz"
    )
    result_files = glob.glob(search_path)

    if not result_files:
        print(f"No result files found at: {search_path}")
        return

    print(f"Found {len(result_files)} result files to analyze.")
    all_summary_data = []
    all_trials_data = []

    for file_path in sorted(result_files):
        try:
            filename = os.path.basename(file_path)
            parts = filename.split("_")
            # Extract scene_name from the path
            scene_name = file_path.split(os.sep)[-3]

            planner_type = parts[0]
            opt_after_planner = parts[2] == "True"
            num_sections = int(parts[4])

            method_name = planner_type.upper()
            if planner_type in ["prm", "rrt"] and opt_after_planner:
                method_name += "+Opt"

            data = np.load(file_path, allow_pickle=True)

            success_rates = data["success_rates"]
            actual_eval_num = len(success_rates)
            if actual_eval_num == 0:
                print(f"Skipping file {filename} because it contains no trial data.")
                continue

            kinematic_rates = data["kinematic_rates"]
            pos_errors = data["pos_errors"]
            rot_errors = data["rot_errors"]
            times = data["times"]
            failure_stats_list = data["failure_stats"]

            # Collect data for each individual trial for the detailed CSV
            for i in range(actual_eval_num):
                trial_data = {
                    "Method": method_name,
                    "Num Sections": num_sections,
                    "scene_name": scene_name,
                    "Trial": i + 1,
                    "Success Rate (%)": success_rates[i],
                    "Kinematic Reachability Rate (%)": kinematic_rates[i],
                    "Position Error (m)": pos_errors[i],
                    "Rotation Error (rad)": rot_errors[i],
                    "Time (s)": times[i],
                }

                if failure_stats_list.size > 0 and i < len(failure_stats_list):
                    trial_failure_stats = failure_stats_list[i]
                    for key, value in trial_failure_stats.items():
                        col_name = key.replace("_", " ").title()
                        if "Rate" in col_name:
                            col_name += " (%)"
                        if "Num " not in col_name and "Total Samples" not in col_name:
                            trial_data[col_name] = value

                all_trials_data.append(trial_data)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not all_trials_data:
        print("No summary data was processed successfully.")
        return

    # Create a DataFrame containing all trial data
    all_trials_df = pd.DataFrame(all_trials_data)

    # Ensure output directory exists
    output_dir = os.path.dirname(detailed_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed CSV file containing all trials
    all_trials_detailed_path = os.path.join(output_dir, "all_trials_detailed.csv")
    all_trials_df.to_csv(all_trials_detailed_path, index=False, float_format="%.4f")
    print(f"\nDetailed data for all trials saved to: {all_trials_detailed_path}")

    # Filter and save successful trials
    successful_trials_df = all_trials_df[
        all_trials_df["Success Rate (%)"] == 100
    ].copy()
    successful_trials_detailed_path = os.path.join(
        output_dir, "successful_trials_detailed.csv"
    )
    successful_trials_df.to_csv(
        successful_trials_detailed_path, index=False, float_format="%.4f"
    )
    print(
        f"Detailed data for successful trials saved to: {successful_trials_detailed_path}"
    )

    # Group by different dimensions and save summary files
    grouping_keys = {
        "method": ["Method"],
        "num_sections": ["Num Sections"],
        "scene_name": ["scene_name"],
    }

    # Helper function to format and save summary
    def save_summary_df(df, group_vars, filename):
        # Arrange column order for better readability
        id_vars = group_vars
        metric_vars = [col for col in df.columns if col not in id_vars]
        desired_order = [
            "Eval Num",
            "Accuracy (%)",
            "Success Rate Mean (%)",
            "Success Rate Std (%)",
            "Kinematic Reachability Rate Mean (%)",
            "Kinematic Reachability Rate Std (%)",
            "Position Error Mean (m)",
            "Position Error Std (m)",
            "Rotation Error Mean (rad)",
            "Rotation Error Std (rad)",
            "Time Mean (s)",
            "Time Std (s)",
        ]
        ordered_metrics = [col for col in desired_order if col in metric_vars]
        remaining_metrics = sorted(
            [col for col in metric_vars if col not in ordered_metrics]
        )
        final_cols = id_vars + ordered_metrics + remaining_metrics
        df = df[final_cols]

        summary_csv_path = os.path.join(output_dir, filename)
        df.to_csv(summary_csv_path, index=False, float_format="%.4f")

        print(f"\n--- Summary grouped by {', '.join(group_vars)} ---")
        print(f"Summary data saved to: {summary_csv_path}")
        print(df.to_string())

    for summary_type, group_vars in grouping_keys.items():
        summary_df = (
            all_trials_df.groupby(group_vars)
            .apply(_calculate_group_summary)
            .reset_index()
        )
        save_summary_df(summary_df, group_vars, f"summary_by_{summary_type}.csv")

    # Create and save overall summary (summary.csv)
    overall_summary_df = (
        all_trials_df.groupby(["Method", "Num Sections"])
        .apply(_calculate_group_summary)
        .reset_index()
    )
    save_summary_df(overall_summary_df, ["Method", "Num Sections"], "summary.csv")

    # Build mapping keyed by "scene:Num Sections" -> list of failed trial numbers
    failed_trials: dict[str, list[int]] = {}
    for trial in all_trials_data:
        if trial.get("Success Rate (%)", 0) < 100:
            scene = trial.get("scene_name")
            num_sections = trial.get("Num Sections")
            try:
                trial_num = int(trial.get("Trial"))
            except Exception:
                continue
            if scene is not None and num_sections is not None:
                key = f"{scene}:{num_sections}"
                failed_trials.setdefault(key, []).append(trial_num)

    # Deduplicate and sort trial lists for each key
    for key, trials in list(failed_trials.items()):
        failed_trials[key] = sorted(set(trials))


    failed_trials_list = []
    for key, trials in failed_trials.items():
        scene, num_sections = key.split(":", 1)
        failed_trials_list.append(
            {
                "scene": scene,
                "Num Sections": num_sections,
                "Failed Trials": ",".join(map(str, trials)),
            }
        )
    return all_trials_data,failed_trials_list


def analyze_results(all_trials_data: list, results_dir: str):
    """
    Analyze trials where all methods were successful.

    This function filters trials to include only those where every method succeeded
    for a given number of sections. It then calculates statistics for these specific
    trials to provide a more direct comparison of method performance under ideal
    conditions.

    Args:
        all_trials_data (list): A list of dictionaries, each representing a single trial.
        results_dir (str): The directory where analysis results will be saved.
    """
    if not all_trials_data:
        print("No trial data provided for common success analysis.")
        return

    trials_df = pd.DataFrame(all_trials_data)

    # Filter for trials with 100% success rate
    successful_trials_df = trials_df[trials_df["Success Rate (%)"] == 100].copy()

    if successful_trials_df.empty:
        print("No trials with 100% success rate found for common success analysis.")
        return

    new_summary_data = []

    # Group by 'Num Sections'
    for num_sections, section_df in successful_trials_df.groupby("Num Sections"):
        methods = section_df["Method"].unique()
        num_methods = len(methods)

        if num_methods < 3:
            continue

        # Find trials that were successful for all methods
        trial_counts = section_df["Trial"].value_counts()
        common_trials_list = trial_counts[trial_counts == num_methods].index.tolist()
        eval_num = len(common_trials_list)
        print(
            f"Num sections: {num_sections}, Num methods: {num_methods}, Common trials: {common_trials_list}, Total common successful trials: {eval_num} \n"
        )

        if eval_num == 0:
            continue

        common_trials_df = section_df[section_df["Trial"].isin(common_trials_list)]

        # Calculate metrics for each method on common trials
        for method in sorted(methods):
            method_df = common_trials_df[common_trials_df["Method"] == method]
            summary_row = {
                "Method": method,
                "Num Sections": num_sections,
                "Eval Num": eval_num,
                "Position Error Mean (m)": method_df["Position Error (m)"].mean(),
                "Position Error Std (m)": method_df["Position Error (m)"].std(),
                "Rotation Error Mean (rad)": method_df["Rotation Error (rad)"].mean(),
                "Rotation Error Std (rad)": method_df["Rotation Error (rad)"].std(),
                "Time Mean (s)": method_df["Time (s)"].mean(),
                "Time Std (s)": method_df["Time (s)"].std(),
            }
            new_summary_data.append(summary_row)

    if not new_summary_data:
        print("No common successful trials found across any section.")
        return

    # Create and save the new summary DataFrame
    new_summary_df = pd.DataFrame(new_summary_data)
    output_filename = "analysis_common_success_summary.csv"
    output_path = os.path.join(results_dir, "analysis", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_summary_df.to_csv(output_path, index=False, float_format="%.4f")

    print(f"\nCommon success analysis summary saved to: {output_path}")
    print("\n--- Common Success Analysis Summary ---")
    print(new_summary_df.to_string())


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
    parser = argparse.ArgumentParser(
        description="Analyze motion planning evaluation results and generate a CSV summary."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/debug_prm_prob/obstacles_random_start_init_True_section_3",
        help="Directory where the .npz result files are stored.",
    )
    parser.add_argument(
        "--detailed_output_name",
        type=str,
        default="all_trials_detailed.csv",
        help="Name for the detailed CSV file with all trial results.",
    )
    args = parser.parse_args()

    detailed_path = os.path.join(
        args.results_dir, "analysis", args.detailed_output_name
    )

    all_trials_data = save_results(
        results_dir=args.results_dir,
        detailed_csv_path=detailed_path,
    )
    # if all_trials_data:
    #     analyze_results(all_trials_data, args.results_dir)
