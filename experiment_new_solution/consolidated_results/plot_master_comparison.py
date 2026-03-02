import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_master_plots(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Standardize model names and order
    df['Model'] = df['Model'].str.replace("_", "-").str.upper()
    model_order = ["TINY-BERT", "MINI-BERT", "MEDIUM-BERT", "DISTIL-BERT", "MINI-LM"]
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values(['Model'])

    sns.set_theme(style="whitegrid")
    metrics = {
        "Val_SST2_Acc": "SST-2 Accuracy",
        "Val_QQP_F1": "QQP F1 Score",
        "Val_STSB_Pearson": "STS-B Pearson"
    }

    # --- 1. Paradigm Comparison (MTL Centralized vs FL) ---
    df_paradigm = df[df['Task_Type'].str.contains("MTL", na=False)].copy()
    for col, title in metrics.items():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_paradigm, x="Model", y=col, hue="Paradigm", palette="muted")
        plt.title(f"Paradigm Comparison (MTL): {title}", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.05)
        for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"comp_paradigm_{col.lower()}.png"))
        plt.close()

    # --- 2. Distribution Impact (IID vs Non-IID in FL-MTL) ---
    df_dist = df[(df['Paradigm'] == "FL") & (df['Task_Type'].str.contains("MTL", na=False))].copy()
    for col, title in metrics.items():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_dist, x="Model", y=col, hue="Distribution", palette="Set2")
        plt.title(f"Data Distribution Impact (FL-MTL): {title}", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.05)
        for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"comp_distribution_{col.lower()}.png"))
        plt.close()

    # --- 3. Single-Task vs MTL (in FL) ---
    df_fl = df[df['Paradigm'] == "FL"].copy()
    for col, title in metrics.items():
        # Aggregate best for Single-Task
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_fl, x="Model", y=col, hue="Task_Type", palette="pastel")
        plt.title(f"Multi-Task Learning Gain (FL): {title}", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.05)
        for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"comp_tasktype_{col.lower()}.png"))
        plt.close()

    # --- 4. Resource Efficiency (GB) ---
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_paradigm, x="Model", y="Resource_Usage", hue="Paradigm", palette="muted")
    plt.title("Resource Consumption: Centralized vs FL (GB)", fontsize=14, fontweight='bold')
    plt.ylabel("Memory (GB)")
    for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comp_resource.png"))
    plt.close()

    print(f"Comprehensive comparison plots generated in: {plots_dir}")

    print(f"All plots generated in: {plots_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv = os.path.join(script_dir, "master_model_comparison.csv")
    create_master_plots(master_csv, script_dir)
