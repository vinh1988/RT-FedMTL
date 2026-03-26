import pandas as pd
from pathlib import Path

# Define paths
input_file = Path('master_model_comparison.csv')
output_dir = Path('latex/plots')
output_file = output_dir / 'all_result.csv'

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Define columns to extract
columns_to_extract = [
    'Paradigm', 'Task_Type', 'Distribution', 'Tuning_Method',
    'Val_SST2_Acc', 'Val_SST2_F1', 
    'Val_QQP_Acc', 'Val_QQP_F1', 
    'Val_STSB_Pearson', 'Val_STSB_Spearman', 
    'Total_Train_Time', 'Resource_Usage', 'Model'
]

def extract_metrics():
    print(f"Reading {input_file}...")
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    df = pd.read_csv(input_file)
    
    # Check if all columns exist
    missing_cols = [col for col in columns_to_extract if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in source file: {missing_cols}")
        return

    # Extract and map
    result_df = df[columns_to_extract].copy()
    
    # Model mapping
    model_mapping = {
        'distil-bert': 'DistilBERT',
        'medium-bert': 'BERT-Medium',
        'mini-bert': 'BERT-Mini',
        'mini-lm': 'MiniLM',
        'tiny_bert': 'TinyBERT'
    }
    result_df['Model'] = result_df['Model'].replace(model_mapping)
    
    # Round numeric columns to 4 digits
    result_df = result_df.round(4)
    
    # Save
    result_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(result_df)} rows to {output_file}")

if __name__ == "__main__":
    extract_metrics()
