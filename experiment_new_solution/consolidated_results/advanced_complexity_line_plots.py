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

# Balanced-enhanced font sizes (Fixed at 15pt+ for readability)
BALANCED_LABELS_SIZE = 27
BALANCED_TITLE_SIZE = 33
BALANCED_LEGEND_SIZE = 24
BALANCED_TICK_SIZE = 22
BALANCED_VALUE_SIZE = 22
BALANCED_XTICK_SIZE = 27

# Publication-quality "Journal" palette (Professional/Muted)
MODEL_STYLES = {
    'DistilBERT': {'color': '#0d47a1', 'marker': 'o'},   # Deep Blue, Circle
    'BERT-Medium': {'color': '#b71c1c', 'marker': 's'},  # Deep Red, Square
    'MiniLM': {'color': '#1b5e20', 'marker': '^'},       # Deep Green, Triangle
    'BERT-Mini': {'color': '#e65100', 'marker': 'D'},    # Deep Orange, Diamond
    'TinyBERT': {'color': '#4a148c', 'marker': 'v'}      # Deep Purple, Reverse Triangle
}

def apply_publication_style():
    """Applies global RC parameters for an academic look"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'axes.labelweight': 'bold',
        'axes.titlesize': 30,
        'axes.labelsize': 24,
        'legend.fontsize': 21,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'figure.dpi': 300
    })

def color_model_legend(ax, legend):
    """Helper to color Legend text by model type for consistency"""
    for text in legend.get_texts():
        label = text.get_text()
        for model, style in MODEL_STYLES.items():
            if model in label:
                text.set_color(style['color'])
                break

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
        # Pivot or format metrics for the MD table
        md_content += "\n| Configuration | " + " | ".join(metrics_data.columns[2:]) + " |\n"
        md_content += "|---|" + "|".join(["---"] * (len(metrics_data.columns) - 2)) + "|\n"
        
        for _, row in metrics_data.iterrows():
            config = f"{row['Model']}-{row['Task']}"
            md_content += f"| {config} | " + " | ".join([f"{val:.4f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val) for val in row[2:]]) + " |\n"
    
    md_content += f"""

## Data Source
- **File**: master_model_comparison.csv
- **Complexity Stages**: 1. Cent Single, 2. Cent Multi, 3. FL Single, 4. FL Multi Non-IID, 5. FL Multi IID, 6. FL Multi LoRA

---
"""
    
    # Save markdown file
    md_path = plots_dir / f"{filename}.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    plt.close(fig)
    return plot_path, md_path

def extract_trajectory_values(m_df, metric_col):
    """Helper to extract metric values across 6 complexity stages"""
    vals = []
    # 1. Cent-Single
    vals.append(m_df[(m_df['Paradigm'] == 'Centralized') & (m_df['Task_Type'] == 'Single-Task')][metric_col].mean())
    # 2. Cent-Multi
    vals.append(m_df[(m_df['Paradigm'] == 'Centralized') & (m_df['Task_Type'] == 'Multi-Task')][metric_col].mean())
    # 3. FL-Single
    vals.append(m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Single-Task') & (m_df['Distribution'] == 'Non-IID')][metric_col].mean())
    # 4. FL-Multi (Non-IID)
    vals.append(m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & (m_df['Distribution'] == 'Non-IID') & (~m_df['Experiment'].str.contains('lora', case=False))][metric_col].mean())
    # 5. FL-Multi (IID)
    vals.append(m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & (m_df['Distribution'] == 'IID') & (~m_df['Experiment'].str.contains('lora', case=False))][metric_col].mean())
    # 6. FL-Multi (LoRA)
    vals.append(m_df[(m_df['Paradigm'] == 'FL') & (m_df['Task_Type'] == 'Multi-Task') & (m_df['Experiment'].str.contains('lora', case=False))][metric_col].mean())
    
    return [v if not pd.isna(v) and v is not None else 0 for v in vals]

def plot_advanced_complexity_line_performance_balanced():
    """Performance trajectory line plot with Journal-quality aesthetics"""
    apply_publication_style()
    
    # Load data
    if not os.path.exists('master_model_comparison.csv'):
        print("❌ master_model_comparison.csv not found!")
        return False
        
    df = pd.read_csv('master_model_comparison.csv')
    df['Model'] = df['Model'].replace(label_mapping)
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    tasks = [('SST2', 'Val_SST2_Acc'), ('QQP', 'Val_QQP_Acc'), ('STSB', 'Val_STSB_Pearson')]
    
    # X-Axis Stages
    stages = [
        'Cent-Single',
        'Cent-Multi',
        'FL-Single',
        'FL-Multi (Non-IID)',
        'FL-Multi (IID)',
        'FL-Multi (LoRA)'
    ]
    
    # Prepare data for subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 11), sharey=True)
    sns.set_style("ticks") # Clean academic ticks
    
    viz_data_list = []
    
    for t_idx, (task_label, acc_col) in enumerate(tasks):
        ax = axes[t_idx]
        
        # Subtle Grayscale Complexity Zones for professional look
        ax.axvspan(-0.5, 1.5, color='#f0f0f0', alpha=0.8, zorder=0)
        ax.axvspan(1.5, 5.5, color='#ffffff', alpha=1.0, zorder=0)
        ax.axvline(1.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.4, zorder=1) # The "Distributed Divide"

        # Baseline horizontal line (Peak centralized performance)
        centralized_peak = df[(df['Paradigm'] == 'Centralized') & (df['Task_Type'] == 'Single-Task')][acc_col].max()
        if not pd.isna(centralized_peak):
            # Sharp Baseline (Single distinct line)
            ax.axhline(y=centralized_peak, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)

        for model in models:
            m_df = df[df['Model'] == model]
            vals = extract_trajectory_values(m_df, acc_col)
            
            # Record for MD table
            row = {'Model': model, 'Task': task_label}
            for i, s in enumerate(stages):
                row[s] = vals[i]
            viz_data_list.append(row)
            
            # Styling for current model
            style = MODEL_STYLES.get(model, {'color': '#333333', 'marker': 'o'})
            
            # Area fill for visual weight
            ax.fill_between(range(len(stages)), vals, color=style['color'], alpha=0.07, zorder=3)
            
            # Main line (Sharp scientific look with distinct markers)
            line = ax.plot(range(len(stages)), vals, 
                         marker=style['marker'], 
                         linewidth=2.5, 
                         markersize=12, 
                         label=model, 
                         color=style['color'], 
                         alpha=0.9, 
                         markeredgecolor='white', 
                         markeredgewidth=1.5, 
                         zorder=5)
            
            # Only label start and end with bold professional text
            for i, val in enumerate(vals):
                if val > 0 and (i == 0 or i == 5):
                    ax.annotate(f'{val:.2f}', 
                               xy=(i, val), 
                               xytext=(0, 12), 
                               textcoords='offset points',
                               ha='center', 
                               fontsize=16, 
                               fontweight='bold', 
                               color=style['color'])

        ax.set_title(f'{task_label} Performance Trajectory', fontsize=22, fontweight='bold', pad=25)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=35, ha='right', fontsize=19)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # Region labels
        ax.text(0.5, 1.02, 'Centralized Paradigms', ha='center', fontsize=12, color='gray', fontstyle='italic', transform=ax.get_xaxis_transform())
        ax.text(3.5, 1.02, 'Federated Learning Environments', ha='center', fontsize=12, color='gray', fontstyle='italic', transform=ax.get_xaxis_transform())

        # Legend (Clean frame)
        leg = ax.legend(fontsize=18, frameon=True, shadow=False, loc='lower left', framealpha=0.9)
        color_model_legend(ax, leg)

    # plt.suptitle('Advanced Performance Comparison: Line View across System Complexity', 
    #              fontsize=BALANCED_TITLE_SIZE + 4, fontweight='bold', y=1.03)
    
    plt.tight_layout()
    
    description = "Trajectory analysis showing how model performance degrades or improves as we move from Centralized ideal states to complex Federated Multi-Task environments."
    insights = """- **Reliability Gap**: Visual drop-off between Centralized and FL configs.
- **MTL Benefit**: Comparison between FL Single and FL Multi Non-IID points.
- **Optimization Impact**: The final point (LoRA) shows recovery in performance despite system complexity."""
    
    viz_df = pd.DataFrame(viz_data_list)
    save_plot_with_metadata(fig, "advanced_complexity_line_performance_balanced", 
                          "Advanced Complexity Performance Trajectory Analysis", description, insights, viz_df)
    
    return True

def plot_advanced_complexity_efficiency_trajectories():
    """Efficiency trajectories (Time and Resources) across 6 complexity stages"""
    apply_publication_style()
    
    # Load data
    if not os.path.exists('master_model_comparison.csv'):
        return False
    df = pd.read_csv('master_model_comparison.csv')
    df['Model'] = df['Model'].replace(label_mapping)
    df['Task_Type'] = df['Task_Type'].replace(label_mapping)
    
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    metrics = [
        ('Training Time', 'Total_Train_Time', 'Time (s)'),
        ('Resource Usage', 'Resource_Usage', 'Relative Units')
    ]
    
    stages = ['Cent-S', 'Cent-M', 'FL-S', 'FL-M (Non-I)', 'FL-M (IID)', 'FL-M (LoRA)']
    
    for title_prefix, col_name, y_label in metrics:
        fig, ax = plt.subplots(figsize=(14, 9))
        sns.set_style("ticks")
        
        # Complexity Zones
        ax.axvspan(-0.5, 1.5, color='#f5f5f5', alpha=1.0, zorder=0)
        ax.axvline(1.5, color='gray', linestyle='-', linewidth=2, alpha=0.3, zorder=1)
        
        viz_data_list = []
        for model in models:
            m_df = df[df['Model'] == model]
            vals = extract_trajectory_values(m_df, col_name)
            
            style = MODEL_STYLES.get(model, {'color': '#333333', 'marker': 'o'})
            
            # Area fill for visual weight
            ax.fill_between(range(len(stages)), vals, color=style['color'], alpha=0.07, zorder=3)
            
            ax.plot(range(len(stages)), vals, marker=style['marker'], linewidth=3, 
                    markersize=12, label=model, color=style['color'], 
                    markeredgecolor='white', zorder=5)
            
            # MD trace
            viz_data_list.append({'Model': model, 'Task': 'Overall', **dict(zip(stages, vals))})
            
            # Start/End labels
            if vals[0] > 0:
                label_0 = f'{vals[0]/1000:.1f}k' if vals[0] > 1000 else f'{vals[0]:.1f}'
                ax.annotate(label_0, xy=(0, vals[0]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=16, fontweight='bold', color=style['color'])
            if vals[5] > 0:
                label_5 = f'{vals[5]/1000:.1f}k' if vals[5] > 1000 else f'{vals[5]:.1f}'
                ax.annotate(label_5, xy=(5, vals[5]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=16, fontweight='bold', color=style['color'])

        # ax.set_title(f'{title_prefix} Scaling Trajectory', fontsize=30, fontweight='bold', pad=25)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=30, ha='right', fontsize=21)
        ax.set_ylabel(y_label, fontsize=24, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        ax.text(0.5, 1.02, 'Centralized Cost', ha='center', fontsize=12, color='gray', fontstyle='italic', transform=ax.get_xaxis_transform())
        ax.text(3.5, 1.02, 'Federated Learning Overhead', ha='center', fontsize=12, color='gray', fontstyle='italic', transform=ax.get_xaxis_transform())
        
        leg = ax.legend(fontsize=19, frameon=True, loc='best')
        color_model_legend(ax, leg)
        
        filename = f"advanced_complexity_line_{col_name.lower()}_balanced"
        save_plot_with_metadata(fig, filename, f"{title_prefix} Trajectory", 
                              f"How {title_prefix.lower()} scales across paradigm complexity.", 
                              "Clear visual jump in cost during FL transition.", pd.DataFrame(viz_data_list))
    return True

if __name__ == "__main__":
    print("🚀 Generating Advanced Complexity Trajectories...")
    perf_ok = plot_advanced_complexity_line_performance_balanced()
    eff_ok = plot_advanced_complexity_efficiency_trajectories()
    
    if perf_ok and eff_ok:
        print("✅ Success! All trajectory plots saved to plots/")
    else:
        print("⚠️ Some plots failed to generate.")
