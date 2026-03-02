import os
import pandas as pd
import glob
from datetime import datetime

def categorize_experiment(exp_name):
    """Parses experiment metadata from folder name."""
    exp_name_lower = exp_name.lower()
    
    # Paradigm
    if "centralized" in exp_name_lower:
        paradigm = "Centralized"
    elif "fl" in exp_name_lower:
        paradigm = "FL"
    else:
        paradigm = "Other"
        
    # Task Type
    if "mtl" in exp_name_lower or "multi-task" in exp_name_lower:
        task_type = "Multi-Task (MTL)"
    else:
        task_type = "Single-Task"
        
    # Distribution
    if "non-iid" in exp_name_lower:
        distribution = "Non-IID"
    else:
        distribution = "IID"
        
    return paradigm, task_type, distribution

def get_best_metric(df, col_names):
    """Checks multiple possible column names and returns the max value."""
    for col in col_names:
        if col in df.columns:
            val = df[col].max()
            if pd.notna(val) and abs(val) > 1e-6:
                return val
    return None

def extract_metrics(root_dir):
    results = []
    
    experiment_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for exp_name in experiment_dirs:
        exp_path = os.path.join(root_dir, exp_name)
        paradigm, task_type, distribution = categorize_experiment(exp_name)
        
        result_files = glob.glob(os.path.join(exp_path, "**", f"*{exp_name}*.csv"), recursive=True)
        result_files = [f for f in result_files if "device_usage" not in f and "training_" not in f]
        
        if not result_files:
            result_files = glob.glob(os.path.join(exp_path, "**", "*.csv"), recursive=True)
            result_files = [f for f in result_files if "device_usage" not in f and "training_" not in f]
            
        if not result_files:
            continue
            
        latest_result_file = max(result_files, key=os.path.getmtime)
        try:
            df_results = pd.read_csv(latest_result_file)
        except Exception as e:
            print(f"Error reading {latest_result_file}: {e}")
            continue
        
        metrics = {
            "Paradigm": paradigm,
            "Task_Type": task_type,
            "Distribution": distribution,
            "Experiment": exp_name
        }

        # --- Performance Metrics ---
        # SST-2: Acc, F1
        metrics["Val_SST2_Acc"] = get_best_metric(df_results, ["global_sst2_val_accuracy", "sst2_val_accuracy", "accuracy"])
        metrics["Val_SST2_F1"] = get_best_metric(df_results, ["global_sst2_val_f1", "sst2_val_f1", "f1_score"])
        
        # QQP: Acc, F1
        metrics["Val_QQP_Acc"] = get_best_metric(df_results, ["global_qqp_val_accuracy", "qqp_val_accuracy"])
        metrics["Val_QQP_F1"] = get_best_metric(df_results, ["global_qqp_val_f1", "qqp_val_f1", "f1_score"])
        
        # STS-B: Pearson, Spearman
        metrics["Val_STSB_Pearson"] = get_best_metric(df_results, ["global_stsb_val_pearson", "stsb_val_pearson", "pearson_correlation"])
        metrics["Val_STSB_Spearman"] = get_best_metric(df_results, ["global_stsb_val_spearman", "stsb_val_spearman", "spearman_correlation"])
        
        metrics["Total_Train_Time"] = df_results.get("training_time", pd.Series([0])).sum()

        # --- Resource Usage (Unified GB Metric) ---
        TOTAL_GPU_MEM = 8.071348224  # RTX 5060 constant
        resource_usage_gb = None

        if paradigm == "Centralized":
            # In centralized results, gpu_usage is the GB allocated
            gpu_allocated_gb = df_results.get("gpu_usage", pd.Series([None]))
            resource_usage_gb = gpu_allocated_gb.mean()
        else:
            # FL paradigm
            device_files = glob.glob(os.path.join(exp_path, "**", "device_usage_*.csv"), recursive=True)
            if device_files:
                df_device = pd.concat([pd.read_csv(f) for f in device_files])
                resource_usage_gb = df_device.get("gpu_memory_allocated_gb", pd.Series([None])).mean()

        metrics["Resource_Usage"] = resource_usage_gb

        results.append(metrics)
        
    return pd.DataFrame(results)

def save_model_summary(summary_df, model_name, output_dir):
    """Saves CSV and Markdown summaries for a specific model."""
    output_model_name = model_name.replace("-", "_")
    output_csv = os.path.join(output_dir, f"{output_model_name}_metrics_summary.csv")
    output_md = os.path.join(output_dir, f"{output_model_name}_metrics_summary.md")
    
    if summary_df.empty:
        print(f"No data for model: {model_name}")
        return

    sort_cols = ["Paradigm", "Task_Type", "Distribution", "Experiment"]
    summary_df = summary_df.sort_values(sort_cols)
    
    # Save full CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary saved to CSV: {output_csv}")
    
    # Markdown summary
    with open(output_md, "w") as f:
        f.write(f"# {model_name.upper()} Comprehensive Metrics Summary\n\n")
        f.write("> [!NOTE]\n")
        f.write("> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.\n\n")
        
        # Task-specific grouped headers for Markdown
        f.write("| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write("| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |\n")
        
        data_cols = [
            "Paradigm", "Task_Type", "Experiment", 
            "Val_SST2_Acc", "Val_SST2_F1", 
            "Val_QQP_Acc", "Val_QQP_F1", 
            "Val_STSB_Pearson", "Val_STSB_Spearman",
            "Total_Train_Time", "Resource_Usage"
        ]
        
        for _, row in summary_df.iterrows():
            vals = []
            for col in data_cols:
                val = row.get(col, None)
                if isinstance(val, (float, int)):
                    if pd.isna(val) or val == 0: vals.append("-")
                    elif abs(val) > 1000: vals.append(f"{val:.2f}")
                    else: vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val) if val is not None else "-")
            f.write("| " + " | ".join(vals) + " |\n")
            
    print(f"Summary saved to Markdown: {output_md}")

def generate_master_comparison(all_results_df, output_dir):
    """Generates a comprehensive master comparison report across all model types."""
    output_md = os.path.join(output_dir, "master_model_comparison.md")
    output_csv = os.path.join(output_dir, "master_model_comparison.csv")
    
    if all_results_df.empty:
        return

    # Sort by Model Size and Paradigm
    model_order = ["tiny_bert", "mini-bert", "medium-bert", "distil-bert", "mini-lm"]
    all_results_df['Model_Name_Lower'] = all_results_df['Model'].str.lower().replace("_", "-")
    
    # Save master CSV
    all_results_df.to_csv(output_csv, index=False)
    print(f"Master summary saved to CSV: {output_csv}")

    with open(output_md, "w") as f:
        f.write("# Master Model Multi-Dimensional Comparison\n\n")
        f.write("This report provides a comprehensive analysis of BERT models across Paradigms, Task Types, and Data Distributions.\n\n")
        
        f.write("## Comparison Tables Overview\n\n")
        f.write("The following benchmark tables are included in this report:\n\n")
        f.write("1.  **Paradigm Performance**: Centralized vs. Federated (MTL)\n")
        f.write("    *   **Attributes**: `Model`, `Paradigm`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`\n")
        f.write("2.  **Data Distribution Impact**: IID vs. Non-IID (FL-MTL)\n")
        f.write("    *   **Attributes**: `Model`, `Distribution`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`\n")
        f.write("3.  **Multi-Task Learning Gain**: Single-Task vs. MTL (FL)\n")
        f.write("    *   **Attributes**: `Model`, `Task Type`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`\n\n")

        # --- SECTION 1: Paradigm Performance ---
        f.write("## 1. Paradigm Performance: Centralized vs. Federated (MTL)\n")
        f.write("Comparing best Multi-Task Learning (MTL) results for each paradigm.\n\n")
        
        f.write("````carousel\n")
        f.write("![Paradigm Comparison: SST-2](./plots/comp_paradigm_val_sst2_acc.png)\n<!-- slide -->\n")
        f.write("![Paradigm Comparison: QQP](./plots/comp_paradigm_val_qqp_f1.png)\n<!-- slide -->\n")
        f.write("![Paradigm Comparison: STS-B](./plots/comp_paradigm_val_stsb_pearson.png)\n<!-- slide -->\n")
        f.write("![Resource Usage: Centralized vs FL](./plots/comp_resource.png)\n")
        f.write("````\n\n")

        f.write("| Model | Paradigm | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for model in model_order:
            model_data = all_results_df[all_results_df['Model_Name_Lower'] == model]
            for paradigm in ["Centralized", "FL"]:
                p_data = model_data[(model_data['Paradigm'] == paradigm) & 
                                    (model_data['Task_Type'].str.contains("MTL"))].copy()
                if not p_data.empty:
                    row = p_data.iloc[0]
                    res = row.get('Resource_Usage', 0)
                    res_str = f"{res:.4f}" if pd.notna(res) else "-"
                    f.write(f"| {model.upper()} | {paradigm} | {row.get('Val_SST2_Acc',0):.4f} | {row.get('Val_QQP_F1',0):.4f} | {row.get('Val_STSB_Pearson',0):.4f} | {res_str} |\n")
            f.write("| | | | | | |\n")
        f.write("\n")

        # --- SECTION 2: Data Distribution Impact ---
        f.write("## 2. Data Distribution Impact: IID vs. Non-IID (FL-MTL)\n")
        f.write("Comparing the effect of Non-IID data on Federated Multi-Task Learning.\n\n")
        
        f.write("````carousel\n")
        f.write("![Distribution Impact: SST-2](./plots/comp_distribution_val_sst2_acc.png)\n<!-- slide -->\n")
        f.write("![Distribution Impact: QQP](./plots/comp_distribution_val_qqp_f1.png)\n<!-- slide -->\n")
        f.write("![Distribution Impact: STS-B](./plots/comp_distribution_val_stsb_pearson.png)\n")
        f.write("````\n\n")

        f.write("| Model | Distribution | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for model in model_order:
            model_data = all_results_df[(all_results_df['Model_Name_Lower'] == model) & (all_results_df['Paradigm'] == 'FL')]
            for dist in ["IID", "Non-IID"]:
                d_data = model_data[(model_data['Distribution'] == dist) & 
                                    (model_data['Task_Type'].str.contains("MTL"))].copy()
                if not d_data.empty:
                    row = d_data.iloc[0]
                    f.write(f"| {model.upper()} | {dist} | {row.get('Val_SST2_Acc',0):.4f} | {row.get('Val_QQP_F1',0):.4f} | {row.get('Val_STSB_Pearson',0):.4f} | {row.get('Resource_Usage',0):.4f} |\n")
            f.write("| | | | | | |\n")
        f.write("\n")

        # --- SECTION 3: Task Type Comparison ---
        f.write("## 3. Multi-Task Learning Gain: Single-Task vs. MTL (FL)\n")
        f.write("Evaluating the performance delta between Single-Task and Multi-Task Federated Learning.\n\n")
        
        f.write("````carousel\n")
        f.write("![Task Type Comparison: SST-2](./plots/comp_tasktype_val_sst2_acc.png)\n<!-- slide -->\n")
        f.write("![Task Type Comparison: QQP](./plots/comp_tasktype_val_qqp_f1.png)\n<!-- slide -->\n")
        f.write("![Task Type Comparison: STS-B](./plots/comp_tasktype_val_stsb_pearson.png)\n")
        f.write("````\n\n")

        f.write("| Model | Task Type | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for model in model_order:
            model_data = all_results_df[(all_results_df['Model_Name_Lower'] == model) & (all_results_df['Paradigm'] == 'FL')]
            for t_type in ["Single-Task", "Multi-Task (MTL)"]:
                t_data = model_data[model_data['Task_Type'] == t_type].copy()
                if not t_data.empty:
                    f.write(f"| {model.upper()} | {t_type} | {t_data['Val_SST2_Acc'].max():.4f} | {t_data['Val_QQP_F1'].max():.4f} | {t_data['Val_STSB_Pearson'].max():.4f} | {t_data['Resource_Usage'].mean():.4f} |\n")
            f.write("| | | | | | |\n")

    print(f"Comprehensive master comparison saved to Markdown: {output_md}")

if __name__ == "__main__":
    # Correct relative paths now that script is in experiment_new_solution/consolidated_results
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_process_output_data"))
    output_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all model directories
    model_dirs = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
    
    print(f"Found {len(model_dirs)} model categories: {model_dirs}")
    
    all_dfs = []
    for model in model_dirs:
        model_path = os.path.join(base_data_dir, model)
        print(f"\n--- Processing Model: {model} ---")
        summary_df = extract_metrics(model_path)
        if not summary_df.empty:
            summary_df['Model'] = model
            save_model_summary(summary_df, model, output_root)
            all_dfs.append(summary_df)
    
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        generate_master_comparison(master_df, output_root)
