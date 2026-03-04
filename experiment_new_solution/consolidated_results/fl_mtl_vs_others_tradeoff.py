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

# Map labels for consistency
label_mapping = {
    'Multi-Task (MTL)': 'Multi-Task',
    'mini-lm': 'MiniLM',
    'tiny-bert': 'TinyBERT',
    'tiny_bert': 'TinyBERT',
    'distil-bert': 'DistilBERT',
    'mini-bert': 'BERT-Mini',
    'medium-bert': 'BERT-Medium'
}

# Publication style
def apply_publication_style():
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

def plot_performance_efficiency_tradeoff():
    """Main method trade-off plot: Performance vs Efficiency"""
    apply_publication_style()
    
    if not os.path.exists('master_model_comparison.csv'):
        print("❌ master_model_comparison.csv not found!")
        return False
        
    df = pd.read_csv('master_model_comparison.csv')
    df['Model'] = df['Model'].replace(label_mapping)
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    
    # Calculate Average Performance (MTL models)
    perf_cols = ['Val_SST2_Acc', 'Val_QQP_Acc', 'Val_STSB_Pearson']
    df['Avg_Performance'] = df[perf_cols].mean(axis=1)
    
    # Define Experimental Categories for Highlighting
    def categorize(row):
        if row['Paradigm'] == 'Centralized':
            return 'Centralized Baseline'
        if row['Paradigm'] == 'FL' and row['Task_Type'] == 'Single-Task':
            return 'FL Single-Task'
        if row['Paradigm'] == 'FL' and 'lora' in str(row['Experiment']).lower():
            return 'FL-MTL (Proposed Main)'
        if row['Paradigm'] == 'FL' and row['Task_Type'] == 'Multi-Task':
            return 'FL-MTL (Full Finetune)'
        return 'Other'

    df['Group'] = df.apply(categorize, axis=1)
    
    # Filter out 'Other' if any
    viz_df = df[df['Group'] != 'Other'].copy()
    
    # Define Styles
    group_styles = {
        'Centralized Baseline': {'color': '#424242', 'marker': 'o', 'alpha': 0.6, 'label': 'Centralized (Upper Bound)'},
        'FL Single-Task': {'color': '#2196F3', 'marker': 's', 'alpha': 0.6, 'label': 'FL Single-Task'},
        'FL-MTL (Full Finetune)': {'color': '#F44336', 'marker': 'D', 'alpha': 0.7, 'label': 'FL-MTL (Standard)'},
        'FL-MTL (Proposed Main)': {'color': '#FFD700', 'marker': '*', 'alpha': 1.0, 'label': 'FL-MTL (Our Main - Efficient)', 'size': 500}
    }

    metrics = [
        ('Total_Train_Time', 'Training Time (s)', 'lower'), # Lower is better
        ('Resource_Usage', 'Relative Resource Usage', 'lower')
    ]

    for col, x_label, goal in metrics:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.set_style("whitegrid", {'axes.grid' : True, 'grid.linestyle': '--'})
        
        # Plot each group
        for group, style in group_styles.items():
            g_df = viz_df[viz_df['Group'] == group]
            if g_df.empty: continue
            
            size = style.get('size', 150)
            edgecolor = 'black' if 'Proposed' in group else 'white'
            linewidth = 2 if 'Proposed' in group else 1
            
            ax.scatter(g_df[col], g_df['Avg_Performance'], 
                       color=style['color'], 
                       marker=style['marker'], 
                       s=size, 
                       alpha=style['alpha'], 
                       label=style['label'],
                       edgecolor=edgecolor,
                       linewidth=linewidth,
                       zorder=5 if 'Proposed' in group else 3)
            
            # Annotate Model Names for clarity
            for _, row in g_df.iterrows():
                ax.annotate(row['Model'], 
                           (row[col], row['Avg_Performance']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8, family='serif')

        # Add Arrows to indicate "Ideal Direction"
        ax.annotate('Higher Performance', xy=(0.05, 0.9), xycoords='axes fraction',
                    xytext=(0.05, 0.7), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    fontsize=12, fontweight='bold', ha='center')
        
        ax.annotate('Lower Cost' if goal == 'lower' else 'Higher Efficiency', 
                    xy=(0.1, 0.05), xycoords='axes fraction',
                    xytext=(0.3, 0.05), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    fontsize=12, fontweight='bold', va='center')

        # Formatting
        ax.set_title(f'Performance vs. {title_map(col)} Trade-off', fontsize=22, fontweight='bold', pad=25)
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax.set_ylabel('Average Accuracy (SST2, QQP, STSB)', fontsize=16, fontweight='bold')
        
        # Legend
        ax.legend(title="Methodology", title_fontsize=14, fontsize=12, loc='lower right', frameon=True, shadow=True)
        
        # Highlight Pareto-like "Sweet Spot"
        ax.text(0.02, 0.98, "Ideal Corner", transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color='darkgreen', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkgreen'))

        filename = f"main_method_tradeoff_{col.lower()}.png"
        plt.tight_layout()
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Generated {filename}")

    return True

def title_map(col):
    return col.replace('_', ' ').title()

if __name__ == "__main__":
    print("🚀 Generating Main Method Trade-off Analysis...")
    if plot_performance_efficiency_tradeoff():
        print("🎉 Trade-off plots successfully generated in plots/")
