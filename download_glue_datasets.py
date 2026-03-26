import os
from datasets import load_dataset

data_dir = "/home/os/Documents/data/code/FedAvgLS/glue_data"

def download_glue_dataset(dataset_name):
    print(f"Downloading {dataset_name} dataset...")
    
    # Load the dataset
    dataset = load_dataset("glue", dataset_name.lower())
    
    # Create dataset directory
    dataset_dir = os.path.join(data_dir, dataset_name.upper())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save each split to TSV files
    for split, data in dataset.items():
        # For some reason, the 'test' split in GLUE doesn't have labels
        if split == 'test' and 'label' not in data.features:
            # Create a version without the label column for test sets
            data = data.remove_columns(['label'])
        
        # Save to TSV
        output_file = os.path.join(dataset_dir, f"{split}.tsv")
        data.to_csv(output_file, sep='\t', index=False)
        print(f"Saved {split} split to {output_file}")

if __name__ == "__main__":
    # Download all three datasets
    for dataset in ["sst2", "qqp", "stsb"]:
        try:
            download_glue_dataset(dataset)
            print(f"Successfully downloaded {dataset}")
        except Exception as e:
            print(f"Error downloading {dataset}: {str(e)}")
    
    print("All datasets have been downloaded and processed.")
