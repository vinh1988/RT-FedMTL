import os
import pandas as pd
import glob
import re
import numpy as np

def detect_sep(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ';' in first_line:
                return ';'
            return ','
    except:
        return ','

def sanitize_header(header):
    return header.strip().lower()

def get_numeric(val):
    try:
        return float(val)
    except:
        return 0.0

def extract_metrics(base_dir):
    summary_results = []
    
    models = ['tiny_bert', 'mini-lm', 'mini-bert', 'medium-bert', 'distil-bert']
    
    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            continue
            
        print(f"Processing model: {model}")
        
        # Traverse categories
        for root, dirs, files in os.walk(model_path):
            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                continue
            
            # Identify experiment folder
            folder_name = os.path.basename(root)
            low_root = root.lower()
            
            # --- 1. Identify Experiment Attributions ---
            is_mtl = 'mtl' in low_root or 'all-tasks' in low_root
            is_fl = 'fl' in low_root or 'federated' in low_root
            is_non_iid = 'non-iid' in low_root
            is_lora = 'lora' in low_root
            
            paradigm = "Multi-task" if is_mtl else "Single-task"
            system = "FL" if is_fl else "Centralized"
            distribution = "Non-IID" if is_non_iid else ("IID" if is_fl else "N/A")
            
            meta_attr = {
                'model': model,
                'experiment': folder_name,
                'paradigm': paradigm,
                'system': system,
                'distribution': distribution,
                'is_lora': is_lora
            }

            # --- 2. Locate Relevant Files ---
            main_file = None
            training_files = {} # task -> filepath
            device_usage_files = []
            
            for f in csv_files:
                f_path = os.path.join(root, f)
                f_low = f.lower()
                
                if "device_usage" in f_low:
                    device_usage_files.append(f_path)
                elif "training_" in f_low and "_metrics_" in f_low:
                    # e.g. training_sst2_metrics...
                    match = re.search(r'training_([a-z0-9]+)_metrics', f_low)
                    if match:
                        training_files[match.group(1)] = f_path
                else:
                    # Potential main results file
                    try:
                        sep = detect_sep(f_path)
                        test_df = pd.read_csv(f_path, sep=sep, nrows=1, engine='python')
                        cols = [sanitize_header(c) for c in test_df.columns]
                        
                        if is_fl:
                            # Federated main files have 'round' and 'global' metrics
                            if 'round' in cols and any('global' in c for c in cols):
                                main_file = f_path
                        else:
                            # Centralized main files have 'epoch' and 'val_accuracy' or 'val_pearson'
                            if 'epoch' in cols and any(c.endswith('_val_accuracy') or c.endswith('_val_pearson') for c in cols):
                                main_file = f_path
                    except:
                        continue
            
            if not main_file:
                continue

            # --- 3. Extract Comprehensive Metrics ---
            try:
                sep = detect_sep(main_file)
                df = pd.read_csv(main_file, sep=sep, engine='python')
                df.columns = [sanitize_header(c) for c in df.columns]
                
                if df.empty:
                    continue

                record = meta_attr.copy()
                
                # GET BEST SCORES (Max for acc/f1/pearson/spearman, Min for loss)
                def get_metric(df, keys, mode='max'):
                    vals = []
                    for k in keys:
                        if k in df.columns:
                            vals.append(get_numeric(df[k].max() if mode == 'max' else df[k].min()))
                    if not vals: return 0.0
                    return max(vals) if mode == 'max' else min(vals)

                # Validation Metrics (Best across all rounds/epochs)
                record['val_sst2_acc'] = get_metric(df, ['global_sst2_val_accuracy', 'sst2_val_accuracy'], 'max')
                record['val_sst2_f1'] = get_metric(df, ['global_sst2_val_f1', 'sst2_val_f1'], 'max')
                record['val_qqp_acc'] = get_metric(df, ['global_qqp_val_accuracy', 'qqp_val_accuracy'], 'max')
                record['val_qqp_f1'] = get_metric(df, ['global_qqp_val_f1', 'qqp_val_f1'], 'max')
                record['val_stsb_pearson'] = get_metric(df, ['global_stsb_val_pearson', 'stsb_val_pearson'], 'max')
                record['val_stsb_spearman'] = get_metric(df, ['global_stsb_val_spearman', 'stsb_val_spearman'], 'max')
                
                # Training Metrics & Samples (Aggregate from device usage or main)
                gpu_peaks = []
                fl_losses = []
                fl_accs = []
                total_samples = 0
                max_sync = 0
                
                if 'synchronization_events' in df.columns:
                    max_sync = get_numeric(df['synchronization_events'].max())

                # Process all device usage files in folder
                for du_file in device_usage_files:
                    try:
                        du_sep = detect_sep(du_file)
                        dudf = pd.read_csv(du_file, sep=du_sep, engine='python')
                        dudf.columns = [sanitize_header(c) for c in dudf.columns]
                        if not dudf.empty:
                            # 1. GPU Memory
                            if 'gpu_memory_allocated_gb' in dudf.columns:
                                gpu_peaks.append(get_numeric(dudf['gpu_memory_allocated_gb'].max()))
                            
                            # 2. FL Loss/Acc (from last round entries)
                            last_round = dudf['round'].max()
                            final_round_entries = dudf[dudf['round'] == last_round]
                            if 'loss' in final_round_entries.columns:
                                fl_losses.append(get_numeric(final_round_entries['loss'].min())) # Best per client
                            if 'accuracy' in final_round_entries.columns:
                                fl_accs.append(get_numeric(final_round_entries['accuracy'].max())) # Best per client
                            
                            # 3. Samples (Sum across all clients for final state)
                            if 'samples_processed' in dudf.columns:
                                total_samples += get_numeric(dudf.iloc[-1]['samples_processed'])
                    except:
                        pass

                if is_fl:
                    record['train_acc'] = np.mean(fl_accs) if fl_accs else get_metric(df, ['avg_accuracy', 'classification_accuracy'], 'max')
                    record['train_loss'] = np.mean(fl_losses) if fl_losses else 0.0
                else:
                    # Centralized - check for joined training metrics
                    train_accs = []
                    train_losses = []
                    for task, t_file in training_files.items():
                        try:
                            t_sep = detect_sep(t_file)
                            tdf = pd.read_csv(t_file, sep=t_sep, engine='python')
                            tdf.columns = [sanitize_header(c) for c in tdf.columns]
                            if not tdf.empty:
                                train_accs.append(get_numeric(tdf['train_accuracy'].max())) # Best train acc
                                train_losses.append(get_numeric(tdf['train_loss'].min())) # Best train loss
                        except:
                            pass
                    
                    record['train_acc'] = np.mean(train_accs) if train_accs else get_metric(df, ['train_accuracy'], 'max')
                    record['train_loss'] = np.mean(train_losses) if train_losses else get_metric(df, ['train_loss', 'total_val_loss'], 'min')

                # For Centralized, also check main file for gpu_usage/samples
                if not is_fl:
                    if 'gpu_usage' in df.columns:
                        gpu_peaks.append(get_numeric(df['gpu_usage'].max()))
                    if 'samples_processed' in df.columns:
                        total_samples = get_numeric(df['samples_processed'].max())
                    
                    # Infer samples for Centralized if still 0
                    if total_samples == 0:
                        task_sizes = {'sst2': 67349, 'qqp': 363846, 'stsb': 5749}
                        if is_mtl:
                            total_samples = sum(task_sizes.values())
                        else:
                            for task, size in task_sizes.items():
                                if task in low_root:
                                    total_samples = size
                                    break

                record['avg_gpu_mem_gb'] = np.mean(gpu_peaks) if gpu_peaks else 0.0
                record['peak_gpu_mem_gb'] = np.max(gpu_peaks) if gpu_peaks else 0.0
                record['total_samples'] = total_samples
                record['sync_events'] = max_sync
                
                # Time consumption
                if 'training_time' in df.columns:
                    record['total_time_sec'] = get_numeric(df['training_time'].sum())
                    record['avg_step_time_sec'] = get_numeric(df['training_time'].mean()) 
                else:
                    record['total_time_sec'] = 0.0
                    record['avg_step_time_sec'] = 0.0

                summary_results.append(record)
                
            except Exception as e:
                import traceback
                print(f"Error processing {folder_name}: {e}")
                traceback.print_exc()

    # --- 4. Save to CSV ---
    if summary_results:
        master_df = pd.DataFrame(summary_results)
        # Ensure consistent column ordering
        cols = ['model', 'experiment', 'paradigm', 'system', 'distribution', 'is_lora',
                'val_sst2_acc', 'val_sst2_f1', 'val_qqp_acc', 'val_qqp_f1', 
                'val_stsb_pearson', 'val_stsb_spearman', 'train_acc', 'train_loss',
                'avg_gpu_mem_gb', 'peak_gpu_mem_gb', 'total_time_sec', 'avg_step_time_sec',
                'total_samples', 'sync_events']
        
        # Keep only existing columns
        master_df = master_df[[c for c in cols if c in master_df.columns]]
        
        output_path = os.path.join(base_dir, "comprehensive_summary_results.csv")
        master_df.to_csv(output_path, index=False)
        print(f"DONE: Comprehensive results (BEST SCORES) saved to {output_path} with {len(master_df)} records.")
    else:
        print("No results found.")

if __name__ == "__main__":
    BASE_DIR = "/home/pqvinh/Documents/LABs/FedAvgLS/experiment_new_solution/pre_process_output_data"
    extract_metrics(BASE_DIR)
