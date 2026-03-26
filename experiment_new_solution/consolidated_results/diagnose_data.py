import pandas as pd
import numpy as np

def diagnose():
    df = pd.read_csv('master_model_comparison.csv')
    print("--- Initial Shapes ---")
    print(f"CSV Rows: {len(df)}")
    print(f"Models in CSV: {df['Model'].unique()}")
    print(f"Task Types in CSV: {df['Task_Type'].unique()}")
    
    label_mapping = {
        'Multi-Task (MTL)': 'Multi-Task',
        'mini-lm': 'MiniLM',
        'tiny-bert': 'TinyBERT',
        'tiny_bert': 'TinyBERT',
        'distil-bert': 'DistilBERT',
        'mini-bert': 'BERT-Mini',
        'medium-bert': 'BERT-Medium'
    }
    
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    df['Model'] = df['Model'].replace(label_mapping)
    
    print("\n--- After Mapping ---")
    print(f"Models: {df['Model'].unique()}")
    print(f"Task Types: {df['Task_Type'].unique()}")
    
    df_filtered = df.dropna(subset=['Model', 'Resource_Usage', 'Paradigm', 'Task_Type', 'Distribution'])
    print(f"\nRows after dropna: {len(df_filtered)}")
    
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    
    centralized_data = df_filtered[df_filtered['Paradigm'] == 'Centralized'].groupby('Model')['Resource_Usage'].mean()
    fl_data = df_filtered[df_filtered['Paradigm'] == 'FL'].groupby('Model')['Resource_Usage'].mean()
    
    print("\n--- Centralized Data Keys ---")
    print(centralized_data.index.tolist())
    
    for model in models:
        val = centralized_data.get(model, 0)
        print(f"Model: {model} -> Centralized Resource Usage: {val}")

if __name__ == "__main__":
    diagnose()
