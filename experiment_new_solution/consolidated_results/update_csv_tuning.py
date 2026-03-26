import pandas as pd

# Load the master comparison file
file_path = '/home/pqvinh/Documents/LABs/FedAvgLS/experiment_new_solution/consolidated_results/master_model_comparison.csv'
df = pd.read_csv(file_path)

# Logic to classify LoRA or FFT based on 'Experiment' name
# From the sample provided:
# - Experiments with 'lora' in the name are marked as 'LoRA'
# - Centralized experiments and others without 'lora' suffix are marked as 'FFT' 
# (assuming centralized ones were full fine-tuned as per typical naming convention in this repo)

def classify_tuning(row):
    exp_name = str(row['Experiment']).lower()
    if 'lora' in exp_name:
        return 'LoRA'
    # Default to FFT for centralized or standard FL unless lora is specified
    return 'FFT'

df['Tuning_Method'] = df.apply(classify_tuning, axis=1)

# Save back to CSV
df.to_csv(file_path, index=False)
print(f"Successfully added Tuning_Method column to {file_path}")
