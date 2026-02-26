#!/usr/bin/env python3
"""
Script to rename all result folders to match the new systematic naming convention.
Based on the results_dir names updated in config files.
"""

import os
import shutil
from pathlib import Path

# Define the base directory
BASE_DIR = Path("c:/Users/hunglq/docs/FedAvgLS/experiment_new_solution/models")

# Mapping of old folder names to new folder names based on config updates
FOLDER_MAPPINGS = {
    # TinyBERT mappings
    "tiny_bert/centralized-mtl-all-tasks/centralized_mtl_results": "tiny_bert/centralized-mtl-all-tasks/tinybert_centralized_mtl",
    "tiny_bert/centralized-single-task-qqp/centralized_qqp_results": "tiny_bert/centralized-single-task-qqp/tinybert_centralized_single_qqp",
    "tiny_bert/centralized-single-task-sst2/centralized_sst2_results": "tiny_bert/centralized-single-task-sst2/tinybert_centralized_single_sst2",
    "tiny_bert/centralized-single-task-stsb/centralized_stsb_results": "tiny_bert/centralized-single-task-stsb/tinybert_centralized_single_stsb",
    "tiny_bert/fl-mtl-slms-berttiny-sts-qqp-sst2/federated_results": "tiny_bert/fl-mtl-slms-berttiny-sts-qqp-sst2/tinybert_fl_mtl_iid_3clients",
    "tiny_bert/fl-mtl-slms-berttiny-sts-qqp-sst2-lora/federated_results": "tiny_bert/fl-mtl-slms-berttiny-sts-qqp-sst2-lora/tinybert_fl_mtl_lora_3clients",
    "tiny_bert/fl-mtl-slms-tiny-bert-non-iid-sst2-qqp-sst2-3client-each/fl_mtl_non-iid-3clients": "tiny_bert/fl-mtl-slms-tiny-bert-non-iid-sst2-qqp-sst2-3client-each/tinybert_fl_mtl_non-iid_9clients",
    "tiny_bert/fl-slms-mini-lm-non-iid-qqp/fl_non_iid_qqp_result": "tiny_bert/fl-slms-mini-lm-non-iid-qqp/tinybert_fl_single_qqp_non-iid_3clients",
    "tiny_bert/fl-slms-mini-lm-non-iid-sst2/fl_non_iid_sts2_result": "tiny_bert/fl-slms-mini-lm-non-iid-sst2/tinybert_fl_single_sst2_non-iid_3clients",
    "tiny_bert/fl-slms-mini-lm-non-iid-stsb/fl_non_iid_stsb_result": "tiny_bert/fl-slms-mini-lm-non-iid-stsb/tinybert_fl_single_stsb_non-iid_3clients",
    
    # DistilBERT mappings
    "distil-bert/centralized-mtl-all-tasks/centralized_mtl_results": "distil-bert/centralized-mtl-all-tasks/distilbert_centralized_mtl",
    "distil-bert/centralized-single-task-qqp/centralized_qqp_results": "distil-bert/centralized-single-task-qqp/distilbert_centralized_single_qqp",
    "distil-bert/centralized-single-task-sst2/centralized_sst2_results": "distil-bert/centralized-single-task-sst2/distilbert_centralized_single_sst2",
    "distil-bert/centralized-single-task-stsb/centralized_stsb_results": "distil-bert/centralized-single-task-stsb/distilbert_centralized_single_stsb",
    "distil-bert/fl-mtl-slms-disti-bert-sts-qqp-sst2/federated_results": "distil-bert/fl-mtl-slms-disti-bert-sts-qqp-sst2/distilbert_fl_mtl_iid_3clients",
    "distil-bert/fl-mtl-slms-disti-bert-sts-qqp-sst2-lora/federated_results": "distil-bert/fl-mtl-slms-disti-bert-sts-qqp-sst2-lora/distilbert_fl_mtl_lora_3clients",
    "distil-bert/fl-mtl-slms-disti-bert-non-iid-sst2-qqp-sst2-3client-each/fl_mtl_non-iid-3clients": "distil-bert/fl-mtl-slms-disti-bert-non-iid-sst2-qqp-sst2-3client-each/distilbert_fl_mtl_non-iid_9clients",
    "distil-bert/fl-slms-mini-lm-non-iid-qqp/fl_non_iid_qqp_result": "distil-bert/fl-slms-mini-lm-non-iid-qqp/distilbert_fl_single_qqp_non-iid_3clients",
    "distil-bert/fl-slms-mini-lm-non-iid-sst2/fl_non_iid_sts2_result": "distil-bert/fl-slms-mini-lm-non-iid-sst2/distilbert_fl_single_sst2_non-iid_3clients",
    "distil-bert/fl-slms-mini-lm-non-iid-stsb/fl_non_iid_stsb_result": "distil-bert/fl-slms-mini-lm-non-iid-stsb/distilbert_fl_single_stsb_non-iid_3clients",
    
    # Medium BERT mappings
    "medium-bert/centralized-mtl-all-tasks/centralized_mtl_results": "medium-bert/centralized-mtl-all-tasks/mediumbert_centralized_mtl",
    "medium-bert/centralized-single-task-qqp/centralized_qqp_results": "medium-bert/centralized-single-task-qqp/mediumbert_centralized_single_qqp",
    "medium-bert/centralized-single-task-sst2/centralized_sst2_results": "medium-bert/centralized-single-task-sst2/mediumbert_centralized_single_sst2",
    "medium-bert/centralized-single-task-stsb/centralized_stsb_results": "medium-bert/centralized-single-task-stsb/mediumbert_centralized_single_stsb",
    "medium-bert/fl-mtl-slms-bertmedium-sts-qqp-sst2/federated_results": "medium-bert/fl-mtl-slms-bertmedium-sts-qqp-sst2/mediumbert_fl_mtl_iid_3clients",
    "medium-bert/fl-mtl-slms-bertmedium-sts-qqp-sst2-lora/federated_results": "medium-bert/fl-mtl-slms-bertmedium-sts-qqp-sst2-lora/mediumbert_fl_mtl_lora_3clients",
    "medium-bert/fl-mtl-slms-medium-bert-non-iid-sst2-qqp-sst2-3client-each/fl_mtl_non-iid-3clients": "medium-bert/fl-mtl-slms-medium-bert-non-iid-sst2-qqp-sst2-3client-each/mediumbert_fl_mtl_non-iid_9clients",
    "medium-bert/fl-slms-mini-lm-non-iid-qqp/fl_non_iid_qqp_result": "medium-bert/fl-slms-mini-lm-non-iid-qqp/mediumbert_fl_single_qqp_non-iid_3clients",
    "medium-bert/fl-slms-mini-lm-non-iid-sst2/fl_non_iid_sts2_result": "medium-bert/fl-slms-mini-lm-non-iid-sst2/mediumbert_fl_single_sst2_non-iid_3clients",
    "medium-bert/fl-slms-mini-lm-non-iid-stsb/fl_non_iid_stsb_result": "medium-bert/fl-slms-mini-lm-non-iid-stsb/mediumbert_fl_single_stsb_non-iid_3clients",
    
    # Mini BERT mappings
    "mini-bert/centralized-mtl-all-tasks/centralized_mtl_results": "mini-bert/centralized-mtl-all-tasks/minibert_centralized_mtl",
    "mini-bert/centralized-single-task-qqp/centralized_qqp_results": "mini-bert/centralized-single-task-qqp/minibert_centralized_single_qqp",
    "mini-bert/centralized-single-task-sst2/centralized_sst2_results": "mini-bert/centralized-single-task-sst2/minibert_centralized_single_sst2",
    "mini-bert/centralized-single-task-stsb/centralized_stsb_results": "mini-bert/centralized-single-task-stsb/minibert_centralized_single_stsb",
    "mini-bert/fl-mtl-slms-bertmini-sts-qqp-sst2/federated_results": "mini-bert/fl-mtl-slms-bertmini-sts-qqp-sst2/minibert_fl_mtl_iid_3clients",
    "mini-bert/fl-mtl-slms-bertmini-sts-qqp-sst2-lora/federated_results": "mini-bert/fl-mtl-slms-bertmini-sts-qqp-sst2-lora/minibert_fl_mtl_lora_3clients",
    "mini-bert/fl-mtl-slms-mini-bert-non-iid-sst2-qqp-sst2-3client-each/fl_mtl_non-iid-3clients": "mini-bert/fl-mtl-slms-mini-bert-non-iid-sst2-qqp-sst2-3client-each/minibert_fl_mtl_non-iid_9clients",
    "mini-bert/fl-slms-mini-lm-non-iid-qqp/fl_non_iid_qqp_result": "mini-bert/fl-slms-mini-lm-non-iid-qqp/minibert_fl_single_qqp_non-iid_3clients",
    "mini-bert/fl-slms-mini-lm-non-iid-sst2/fl_non_iid_sts2_result": "mini-bert/fl-slms-mini-lm-non-iid-sst2/minibert_fl_single_sst2_non-iid_3clients",
    "mini-bert/fl-slms-mini-lm-non-iid-stsb/fl_non_iid_stsb_result": "mini-bert/fl-slms-mini-lm-non-iid-stsb/minibert_fl_single_stsb_non-iid_3clients",
    
    # MiniLM mappings
    "mini-lm/centralized-mtl-all-tasks/centralized_mtl_results": "mini-lm/centralized-mtl-all-tasks/minilm_centralized_mtl",
    "mini-lm/centralized-single-task-qqp/centralized_qqp_results": "mini-lm/centralized-single-task-qqp/minilm_centralized_single_qqp",
    "mini-lm/centralized-single-task-sst2/centralized_sst2_results": "mini-lm/centralized-single-task-sst2/minilm_centralized_single_sst2",
    "mini-lm/centralized-single-task-stsb/centralized_stsb_results": "mini-lm/centralized-single-task-stsb/minilm_centralized_single_stsb",
    "mini-lm/fl-mtl-slms-mini-lm-sts-qqp-sst2/fl_mtl_iid_1client": "mini-lm/fl-mtl-slms-mini-lm-sts-qqp-sst2/minilm_fl_mtl_iid_3clients",
    "mini-lm/fl-mtl-slms-mini-lm-sts-qqp-sst2-lora/federated_results": "mini-lm/fl-mtl-slms-mini-lm-sts-qqp-sst2-lora/minilm_fl_mtl_lora_3clients",
    "mini-lm/fl-mtl-slms-mini-lm-non-iid-sst2-qqp-sst2-3client-each/fl_mtl_non-iid-3clients": "mini-lm/fl-mtl-slms-mini-lm-non-iid-sst2-qqp-sst2-3client-each/minilm_fl_mtl_non-iid_9clients",
    "mini-lm/fl-slms-mini-lm-non-iid-qqp/fl_non_iid_qqp_result": "mini-lm/fl-slms-mini-lm-non-iid-qqp/minilm_fl_single_qqp_non-iid_3clients",
    "mini-lm/fl-slms-mini-lm-non-iid-sst2/fl_non_iid_sts2_result": "mini-lm/fl-slms-mini-lm-non-iid-sst2/minilm_fl_single_sst2_non-iid_3clients",
    "mini-lm/fl-slms-mini-lm-non-iid-stsb/fl_non_iid_stsb_result": "mini-lm/fl-slms-mini-lm-non-iid-stsb/minilm_fl_single_stsb_non-iid_3clients",
}

def rename_result_folders():
    """Rename all result folders to match the new naming convention."""
    
    print("🔄 Starting result folder renaming process...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📋 Total folders to process: {len(FOLDER_MAPPINGS)}")
    print()
    
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    for old_path, new_path in FOLDER_MAPPINGS.items():
        old_full_path = BASE_DIR / old_path
        new_full_path = BASE_DIR / new_path
        
        try:
            if old_full_path.exists():
                if not new_full_path.exists():
                    print(f"📂 Renaming: {old_path}")
                    print(f"   └─ To: {new_path}")
                    
                    # Create parent directory if it doesn't exist
                    new_full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Rename the folder
                    shutil.move(str(old_full_path), str(new_full_path))
                    renamed_count += 1
                    print(f"   ✅ Successfully renamed")
                else:
                    print(f"⚠️  Skipping: {new_path} already exists")
                    skipped_count += 1
            else:
                print(f"⚠️  Skipping: {old_path} does not exist")
                skipped_count += 1
                
        except Exception as e:
            print(f"❌ Error renaming {old_path}: {str(e)}")
            error_count += 1
        
        print()
    
    print("📊 SUMMARY:")
    print(f"   ✅ Successfully renamed: {renamed_count}")
    print(f"   ⚠️  Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📋 Total processed: {len(FOLDER_MAPPINGS)}")
    
    if error_count == 0:
        print("\n🎉 All result folders have been successfully renamed!")
    else:
        print(f"\n⚠️  Completed with {error_count} errors. Please check the output above.")

if __name__ == "__main__":
    rename_result_folders()
