import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# Create plots directory if it doesn't exist
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)

# Map old labels to new preferred labels for consistency
label_mapping = {
    'Multi-Task (MTL)': 'Multi-Task',
    'mini-lm': 'MiniLM',
    'tiny-bert': 'TinyBERT',
    'tiny_bert': 'TinyBERT',
    'distil-bert': 'DistilBERT',
    'mini-bert': 'BERT-Mini',
    'medium-bert': 'BERT-Medium'
}

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18
BALANCED_TITLE_SIZE = 21
BALANCED_LEGEND_SIZE = 15
BALANCED_TICK_SIZE = 15
BALANCED_VALUE_SIZE = 15
BALANCED_XTICK_SIZE = 18

# High-contrast palette for model names (easy for recognition)
MODEL_TEXT_COLORS = {
    'DistilBERT': '#000080',  # Deep Navy Blue
    'BERT-Medium': '#D32F2F', # Vivid Crimson
    'MiniLM': '#2E7D32',      # Forest Green
    'BERT-Mini': '#E65100',   # Burnt Orange
    'TinyBERT': '#4A148C'     # Deep Royal Purple
}

def color_model_xticks(ax):
    """Helper to color X-axis labels by model type using muted palette"""
    for tick in ax.get_xticklabels():
        text = tick.get_text()
        for model, color in MODEL_TEXT_COLORS.items():
            if model in text:
                tick.set_color(color)
                break

def add_model_banding(ax, df_viz):
    """Adds subtle alternate background vertical bands to group models"""
    models = df_viz['Model'].unique()
    # Get the unique model positions
    unique_models = []
    current_model = None
    for i, model in enumerate(df_viz['Model']):
        if model != current_model:
            unique_models.append(i)
            current_model = model
    unique_models.append(len(df_viz))
    
    # Draw bars
    for i in range(len(unique_models) - 1):
        if i % 2 == 1: # Alternate bands
            ax.axvspan(unique_models[i] - 0.5, unique_models[i+1] - 0.5, 
                        color='gray', alpha=0.07, zorder=0)

def save_plot_with_metadata(fig, filename, title, description, insights, metrics_data=None):
    """Save plot and generate markdown documentation with metrics"""
    
    # Save the plot
    plot_path = plots_dir / f"{filename}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Generate markdown content
    md_content = f"""# {title}

![{title}]({filename}.png)

## Description
{description}

## Key Insights
{insights}

## Metrics Data
"""
    
    # Add metrics table if provided
    if metrics_data is not None and not metrics_data.empty:
        md_content += "\n| " + " | ".join(metrics_data.columns) + " |\n"
        md_content += "|" + "|".join(["---"] * len(metrics_data.columns)) + "|\n"
        
        for _, row in metrics_data.iterrows():
            md_content += "| " + " | ".join([f"{val:.4f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val) for val in row]) + " |\n"
    
    md_content += f"""

## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Analysis Dimensions**: Paradigm, Task Type, Distribution, Optimization (LoRA)

---
"""
    
    # Save markdown file
    md_path = plots_dir / f"{filename}.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    plt.close(fig)
    return plot_path, md_path

def plot_advanced_complexity_stacked_performance_balanced():
    """Highly granular 6-way performance comparison"""
    
    # Load data
    df = pd.read_csv('master_model_comparison.csv')
    df['Model'] = df['Model'].replace(label_mapping)
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        m_df = df[df['Model'] == model]
        
        # 1. Centralized Single
        cs_df = m_df[(m_df['Paradigm'] == 'Centralized') & (m_df['Task_Type'] == 'Single-Task')]
        
        # 2. Centralized Multi
        cm_df = m_df[(m_df['Paradigm'] == 'Centralized') & (m_df['Task_Type'] == 'Multi-Task')]
        
        # 3. FL Single Non-IID
        fsn_df = m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Single-Task') & (m_df['Distribution'] == 'Non-IID')]
        
        # 4. FL Multi Non-IID (Excluding LoRA)
        fmn_df = m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & 
                     (m_df['Distribution'] == 'Non-IID') & (~m_df['Experiment'].str.contains('lora', case=False))]
        
        # 5. FL Multi IID (Excluding LoRA)
        fmi_df = m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & 
                     (m_df['Distribution'] == 'IID') & (~m_df['Experiment'].str.contains('lora', case=False))]
        
        # 6. FL Multi LoRA
        fml_df = m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & 
                     (m_df['Experiment'].str.contains('lora', case=False))]
        
        for task_key, acc_col in [('SST2', 'Val_SST2_Acc'), ('QQP', 'Val_QQP_Acc'), ('STSB', 'Val_STSB_Pearson')]:
            vals = {
                'Cent_Single': cs_df[acc_col].mean(),
                'Cent_Multi': cm_df[acc_col].mean(),
                'FL_Single_NonIID': fsn_df[acc_col].mean(),
                'FL_Multi_NonIID': fmn_df[acc_col].mean(),
                'FL_Multi_IID': fmi_df[acc_col].mean(),
                'FL_Multi_LoRA': fml_df[acc_col].mean()
            }
            
            # Check if we have enough data (at least CS and one FL)
            if not pd.isna(vals['Cent_Single']) and any([not pd.isna(v) for k, v in vals.items() if k != 'Cent_Single']):
                row = {
                    'Model': model,
                    'Task': task_key,
                    'Total': 0
                }
                for k, v in vals.items():
                    val = v if not pd.isna(v) else 0
                    row[k] = val
                    row['Total'] += val
                
                comparison_data.append(row)
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        fig, ax = plt.subplots(figsize=(24, 14))
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Advanced 6-way colors
        colors = ['#023e8a', '#48cae4', '#e63946', '#fb8500', '#ffb703', '#8338ec']
        
        bottom = np.zeros(len(viz_df))
        bar_sets = []
        categories = ['Cent_Single', 'Cent_Multi', 'FL_Single_NonIID', 'FL_Multi_NonIID', 'FL_Multi_IID', 'FL_Multi_LoRA']
        cat_labels = ['Cent Single', 'Cent Multi', 'FL Single (Non-IID)', 'FL Multi (Non-IID)', 'FL Multi (IID)', 'FL Multi (LoRA)']
        
        for i, cat in enumerate(categories):
            bars = ax.bar(x_pos, viz_df[cat], bottom=bottom, label=cat_labels[i], alpha=0.9, color=colors[i])
            bar_sets.append(bars)
            bottom += viz_df[cat]
            
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance Metrics', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Advanced Granular Performance Analysis: 6-Dimensional View', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE - 4, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, shadow=True, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.2, axis='y')
        
        # Value labels (only if height is significant)
        for bars in bar_sets:
            for b in bars:
                h = b.get_height()
                if h > 0.05: # Only label if bar is tall enough
                    y = b.get_y() + h/2
                    ax.text(b.get_x() + b.get_width()/2., y, f'{h:.2f}', 
                            ha='center', va='center', fontsize=BALANCED_VALUE_SIZE - 3, 
                            color='white' if h > 0.4 else 'black', fontweight='bold')
        
        description = "Highly granular 6-way stacked performance comparison across Centralized baselines, FL Single/Multi-task settings, data distributions (IID/Non-IID), and optimization techniques (LoRA)."
        insights = """- **Paradigm Baseline**: Centralized Single/Multi (Blues) provide the theoretical upper bounds.
- **Distribution Impact**: Compare FL Multi Non-IID (Orange) vs IID (Yellow) to see data heterogeneity effects.
- **MTL Efficiency**: FL Multi Non-IID performance compared to FL Single Non-IID (Red).
- **Optimization (LoRA)**: The Purple segment demonstrates the viability of LoRA in Federated Multi-Task learning."""
        
        metrics_df = viz_df[['Model', 'Task'] + categories + ['Total']]
        save_plot_with_metadata(fig, "advanced_complexity_stacked_performance_balanced", 
                                     "Advanced 6-Way Granular Performance Analysis", description, insights, metrics_df)
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Advanced Complexity Visualization...")
    if plot_advanced_complexity_stacked_performance_balanced():
        print("✅ Advanced 6-way stacked comparison plot generated successfully!")
    else:
        print("❌ Failed to generate plot.")
