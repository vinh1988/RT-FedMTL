import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18
BALANCED_TITLE_SIZE = 22
BALANCED_LEGEND_SIZE = 17
BALANCED_TICK_SIZE = 13
BALANCED_VALUE_SIZE = 13
BALANCED_XTICK_SIZE = 16

# Publication-quality "Journal" palette
CATEGORY_STYLES = {
    'Centralized': {'color': '#0d47a1', 'marker': 'o'}, # Deep Blue
    'FL': {'color': '#b71c1c', 'marker': 's'},          # Deep Red
    'IID': {'color': '#1b5e20', 'marker': '^'},         # Deep Green
    'Non-IID': {'color': '#e65100', 'marker': 'v'},     # Deep Orange
    'Single': {'color': '#4a148c', 'marker': 'D'},      # Deep Purple
    'Multi': {'color': '#333333', 'marker': 'p'}        # Charcoal
}

def apply_publication_style():
    """Applies global RC parameters for an academic look"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'axes.labelweight': 'bold',
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': 300
    })

def color_model_xticks(ax):
    """Helper to color X-axis labels consistently"""
    # Simply ensure bold professional look
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

def add_model_banding(ax, models_list):
    """Adds subtle alternate background vertical bands to group models"""
    # For line plots, models are on the x-axis directly
    for i in range(len(models_list)):
        if i % 2 == 1: # Alternate bands
            ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.07, zorder=0)

def load_and_process_data():
    """Load and process the master model comparison data for performance"""
    df = pd.read_csv('master_model_comparison.csv')
    
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
    
    # Apply mapping to Model and Task_Type columns
    df['Model'] = df['Model'].replace(label_mapping)
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    
    # Calculate Average Performance score
    # We use Acc for SST2 and QQP, and Pearson for STSB
    # Handle NaNs by ignoring them in the average calculation per row
    perf_cols = ['Val_SST2_Acc', 'Val_QQP_Acc', 'Val_STSB_Pearson']
    df['Performance_Score'] = df[perf_cols].mean(axis=1)
    
    # Filter out any rows with missing critical data
    df = df.dropna(subset=['Model', 'Performance_Score', 'Paradigm', 'Task_Type', 'Distribution'])
    
    return df

def prepare_combined_perf_data(df):
    """Aggregate data by Model and various categories for performance comparison"""
    
    # Paradigm breakdown: Centralized vs FL
    paradigm_data = df.groupby(['Model', 'Paradigm'])['Performance_Score'].mean().reset_index()
    paradigm_data.columns = ['Model', 'Category', 'Performance']
    
    # Distribution breakdown: IID vs Non-IID
    distribution_data = df.groupby(['Model', 'Distribution'])['Performance_Score'].mean().reset_index()
    distribution_data.columns = ['Model', 'Category', 'Performance']
    
    # Task Type breakdown: Single vs Multi
    task_data = df.groupby(['Model', 'Task_Type'])['Performance_Score'].mean().reset_index()
    task_data.columns = ['Model', 'Category', 'Performance']
    task_data['Category'] = task_data['Category'].replace({'Single-Task': 'Single', 'Multi-Task': 'Multi'})
    
    # Combine all comparisons
    combined = pd.concat([paradigm_data, distribution_data, task_data])
    
    return combined

def create_performance_line_plot(df, output_path):
    """Create a premium 1x3 grouped line plot for model performance"""
    
    # Prepare combined data
    combined_data = prepare_combined_perf_data(df)
    
    # Get models in order
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    x_pos = np.arange(len(models))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)
    
    apply_publication_style()
    
    # Define comparison groups for subplots
    groups = [
        {'name': 'Paradigm Analysis', 'categories': ['Centralized', 'FL']},
        {'name': 'Distribution Analysis', 'categories': ['IID', 'Non-IID']},
        {'name': 'Task Analysis', 'categories': ['Single', 'Multi']}
    ]
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        for category in group['categories']:
            # Get data for this category
            category_data = combined_data[combined_data['Category'] == category]
            values = np.array([category_data[category_data['Model'] == model]['Performance'].iloc[0] 
                     if len(category_data[category_data['Model'] == model]) > 0 else 0 
                     for model in models])
            
            # Style configuration
            style = CATEGORY_STYLES[category]
            
            # Add subtle area fill for visual weight
            ax.fill_between(x_pos, values, alpha=0.08, color=style['color'], zorder=2)
            
            # Plot main line (Sharp scientific series)
            ax.plot(x_pos, values, 
                    color=style['color'], 
                    marker=style['marker'], 
                    linewidth=3, 
                    markersize=14,
                    markeredgewidth=2,
                    markeredgecolor='white',
                    label=category,
                    alpha=0.9,
                    zorder=3)
            
            # Add value labels
            for i, (pos, val) in enumerate(zip(x_pos, values)):
                if val > 0:
                    ax.text(pos, val + 0.015, f'{val:.3f}', ha='center', va='bottom', 
                            fontsize=BALANCED_VALUE_SIZE, fontweight='bold',
                            color=style['color'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
        # Subplot customization
        ax.set_title(group['name'], fontsize=BALANCED_TITLE_SIZE, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE - 2, fontweight='bold')
        add_model_banding(ax, models) # Apply model banding
        color_model_xticks(ax) # Apply muted coloring
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, loc='lower left')
        
        if idx == 0:
            ax.set_ylabel('Avg. Performance Score', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')

    plt.suptitle('Comprehensive Model Performance Analysis: Grouped Comparison', 
                 fontsize=BALANCED_TITLE_SIZE + 4, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_data

def save_perf_metrics_data(pivot_data, output_path):
    """Save the aggregated performance metrics to a markdown file"""
    md_content = """# Line Plot Performance Analysis
    
![Line Plot Performance Analysis](line_performance_balanced.png)

## Description
Comprehensive performance comparison using grouped line plots to show trends across all six categories (Centralized, FL, IID, Non-IID, Single, Multi). This visualization uses an average performance score derived from GLUE task metrics.

## Performance Metrics Data

| Model | Centralized | FL | IID | Non-IID | Single | Multi |
|---|---|---|---|---|---|---|
"""
    
    for model in ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']:
        if model in pivot_data.index:
            row = pivot_data.loc[model]
            md_content += f"| {model} | {row['Centralized']:.4f} | {row['FL']:.4f} | {row['IID']:.4f} | {row['Non-IID']:.4f} | {row['Single']:.4f} | {row['Multi']:.4f} |\n"
            
    md_content += """
## Data Source
- **File**: master_model_comparison.csv
- **Models**: DistilBERT, BERT-Medium, BERT-Mini, MiniLM, TinyBERT
- **Metric**: Average of Validation SST2/QQP Acc and STSB Pearson
"""
    
    with open(output_path, 'w') as f:
        f.write(md_content)

def main():
    print("Creating premium performance line plot...")
    
    # Load and process data
    df = load_and_process_data()
    
    # Create the grouped line plot
    output_plot = 'plots/line_performance_balanced.png'
    combined_data = create_performance_line_plot(df, output_plot)
    
    # Prepare pivot data for markdown table
    pivot_data = combined_data.pivot(index='Model', columns='Category', values='Performance')
    
    # Save the metrics to markdown
    output_md = 'plots/line_performance_balanced.md'
    save_perf_metrics_data(pivot_data, output_md)
    
    print(f"✅ Performance line plot created successfully!")
    print(f"📊 Plot saved as: {output_plot}")
    print(f"📄 Metrics saved as: {output_md}")

if __name__ == "__main__":
    main()
