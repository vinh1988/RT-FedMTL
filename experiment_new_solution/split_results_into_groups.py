import pandas as pd
import os

def split_summary_results(csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    
    # 1. System & Paradigm Comparison
    # Cent vs FL vs FL-MTL (MTL context)
    system_comp = df[df['paradigm'] == 'Multi-task']
    system_comp.to_csv(os.path.join(output_dir, 'comp_system_mtl.csv'), index=False)
    
    # STL vs MTL (Across all systems)
    paradigm_comp = df.copy()
    paradigm_comp.to_csv(os.path.join(output_dir, 'comp_paradigm_stl_mtl.csv'), index=False)

    # 2. Distribution Impact (IID vs Non-IID in FL)
    dist_comp = df[df['system'] == 'FL']
    dist_comp.to_csv(os.path.join(output_dir, 'comp_distribution_iid_noniid.csv'), index=False)

    # 3. Optimization (LoRA vs Full Fine-tuning)
    # Filter for experiments where we have both LoRA and non-LoRA counterparts
    lora_comp = df[df['experiment'].str.contains('lora') | df['is_lora'] == True | (df['system'] == 'FL')]
    lora_comp.to_csv(os.path.join(output_dir, 'comp_optimization_lora.csv'), index=False)

    # 4. Model Efficiency Perspective
    eff_df = df.copy()
    # Create a simplified efficiency view
    val_cols = [c for c in df.columns if c.startswith('val_')]
    eff_cols = ['model', 'experiment', 'system', 'paradigm', 'is_lora'] + val_cols + ['avg_gpu_mem_gb', 'peak_gpu_mem_gb', 'total_time_sec', 'total_samples']
    eff_df[eff_cols].to_csv(os.path.join(output_dir, 'comp_model_efficiency.csv'), index=False)

    # 5. Model-specific deep dives
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_name = str(model).replace('-', '_')
        model_df.to_csv(os.path.join(output_dir, f'model_{model_name}_full.csv'), index=False)

    # 6. Legacy groups (keep for compatibility)
    df[(df['system'] == 'FL') & (df['paradigm'] == 'Multi-task')].to_csv(os.path.join(output_dir, 'group_fl_mtl.csv'), index=False)
    df[(df['system'] == 'Centralized') & (df['paradigm'] == 'Multi-task')].to_csv(os.path.join(output_dir, 'group_centralized_mtl.csv'), index=False)

    print(f"Exhaustive CSVs generated in {output_dir}")

    print(f"Master CSV split into {len(df['model'].unique()) + 4} smaller files in {output_dir}")

if __name__ == "__main__":
    master_csv = "/home/pqvinh/Documents/LABs/FedAvgLS/experiment_new_solution/pre_process_output_data/comprehensive_summary_results.csv"
    target_dir = "/home/pqvinh/Documents/LABs/FedAvgLS/experiment_new_solution/plots_comparison"
    split_summary_results(master_csv, target_dir)
