import argparse
import glob
import os

import numpy as np
import pandas as pd


def save_results(results_dir: str, detailed_csv_path: str):
    # Search for the aggregated results files, not individual trial files.
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
            planner_type = parts[0]
            num_sections = int(parts[2])

            data = np.load(file_path, allow_pickle=True)

            # Basic Metrics from the aggregated file
            # These are now arrays of results from all trials for a given config.
            success_rates = data["success_rates"]
            kinematic_rates = data["kinematic_rates"]
            pos_errors = data["pos_errors"]
            rot_errors = data["rot_errors"]
            times = data["times"]
            failure_stats_list = data["failure_stats"]

            actual_eval_num = len(success_rates)
            if actual_eval_num == 0:
                print(f"Skipping {filename} as it contains no trials.")
                continue

            # Collect data for each individual trial for the detailed CSV
            for i in range(actual_eval_num):
                trial_data = {
                    "Method": planner_type.upper(),
                    "Num Sections": num_sections,
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
                        # Clean up key for column name
                        col_name = key.replace("_", " ").title()
                        if "Rate" in col_name:
                            col_name += " (%)"
                        # Exclude raw counts from the detailed table for clarity
                        if "Num " not in col_name and "Total Samples" not in col_name:
                            trial_data[col_name] = value
                all_trials_data.append(trial_data)

            # Calculate aggregated summary
            summary = {
                "Method": planner_type.upper(),
                "Num Sections": num_sections,
                "Eval Num": actual_eval_num,
                "Success Rate Mean (%)": np.nanmean(success_rates),
                "Success Rate Std (%)": np.nanstd(success_rates),
                "Kinematic Reachability Rate Mean (%)": np.nanmean(kinematic_rates),
                "Kinematic Reachability Rate Std (%)": np.nanstd(kinematic_rates),
                "Position Error Mean (m)": np.nanmean(pos_errors),
                "Position Error Std (m)": np.nanstd(pos_errors),
                "Rotation Error Mean (rad)": np.nanmean(rot_errors),
                "Rotation Error Std (rad)": np.nanstd(rot_errors),
                "Time Mean (s)": np.nanmean(times),
                "Time Std (s)": np.nanstd(times),
            }

            # Detailed Failure Analysis for Summary
            if failure_stats_list.size > 0:
                failure_df = pd.DataFrame(list(failure_stats_list))
                # drop columns that are not rates
                failure_df = failure_df.drop(
                    columns=["total_samples", "num_success", "num_fail"],
                    errors="ignore",
                )
                failure_means = failure_df.mean()

                for key, value in failure_means.items():
                    # Add "Mean" and "(%)" to the column names for clarity
                    col_name = (
                        key.replace("_", " ").title().replace("Rate", "Rate Mean")
                    )
                    if "Rate" in col_name:
                        col_name += " (%)"
                    summary[col_name] = value

            all_summary_data.append(summary)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Save Detailed All-Trials CSV
    if not all_trials_data:
        print("No trial data was processed successfully.")
    else:
        detailed_df = pd.DataFrame(all_trials_data)
        os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)
        detailed_df.to_csv(detailed_csv_path, index=False, float_format="%.4f")
        print(f"\nDetailed all-trials data saved to: {detailed_csv_path}")

    # Save Aggregated Summary CSV
    if not all_summary_data:
        print("No summary data was processed successfully.")
        return

    summary_df = pd.DataFrame(all_summary_data)

    # Reorder columns for better readability
    cols_order = [
        "Method",
        "Num Sections",
        "Eval Num",
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
    failure_cols = [col for col in summary_df.columns if col not in cols_order]
    final_cols = cols_order + sorted(failure_cols)
    final_cols_exist = [col for col in final_cols if col in summary_df.columns]
    summary_df = summary_df[final_cols_exist]

    print("\n--- Results Summary ---")
    print(summary_df.to_string())
    return all_trials_data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze motion planning evaluation results and generate a CSV summary."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/13.pick_from_shelf",
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
    if all_trials_data:
        analyze_results(all_trials_data, args.results_dir)
