import os
import json
import pandas as pd


def integrate_results(base_dir, integrated_dir):
    runs = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not runs:
        raise ValueError(f"No run_* directories found in {base_dir}")

    example_run_metrics_dir = os.path.join(base_dir, runs[0], "test_metrics")
    metric_files = [
        f for f in os.listdir(example_run_metrics_dir)
        if f.endswith(".json")
    ]

    os.makedirs(integrated_dir, exist_ok=True)

    for metric_file in metric_files:
        metrics_list = []

        for run in runs:
            file_path = os.path.join(base_dir, run, "test_metrics", metric_file)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}: {type(data)}")

            metrics_list.append(df)

        if not metrics_list:
            continue

        combined_metrics = pd.concat(metrics_list, axis=0, ignore_index=True)

        agg_data = {}
        for col in combined_metrics.columns:
            agg_data[f"av_{col}"] = [combined_metrics[col].mean()]
            agg_data[f"stdev_{col}"] = [combined_metrics[col].std()]

        aggregated_df = pd.DataFrame(agg_data)

        output_file = os.path.join(
            integrated_dir,
            metric_file.replace(".json", "_integrated.csv")
        )
        aggregated_df.to_csv(output_file, index=False)


# Change the dataset name here:
dataset_name = "MNIST_75sp" 
base_dir = f"../results/{dataset_name}"
integrated_dir = f"../results/{dataset_name}/integrated"
os.makedirs(integrated_dir, exist_ok=True)
integrate_results(base_dir, integrated_dir)