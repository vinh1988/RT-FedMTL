import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('master_model_comparison.csv')

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

# Apply mapping to Model and Task_Type columns
df['Model'] = df['Model'].replace(label_mapping)
df['Task_Type'] = df['Task_Type'].replace(label_mapping)

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18   # was 12, now 1.5x
BALANCED_TITLE_SIZE = 21      # was 14, now 1.5x  
BALANCED_LEGEND_SIZE = 17    # was 11, now 1.5x
BALANCED_TICK_SIZE = 15       # was 8, now 1.5x
BALANCED_VALUE_SIZE = 15      # was 8, now 1.5x
BALANCED_XTICK_SIZE = 18      # was 12, now 1.5x for model names

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
- **Total Experiments**: {len(df)}
- **Models**: {', '.join(df['Model'].unique())}
- **Paradigms**: {', '.join(df['Paradigm'].unique())}
- **Task Types**: {', '.join(df['Task_Type'].unique())}
- **Distributions**: {', '.join(df['Distribution'].unique())}

---
"""
    
    # Save markdown file
    md_path = plots_dir / f"{filename}.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    plt.close(fig)
    return plot_path, md_path

def plot_centralized_vs_fl_stacked_performance_balanced():
    """performance comparison: Centralized vs FL"""
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        centralized = model_data[model_data['Paradigm'] == 'Centralized']
        fl = model_data[model_data['Paradigm'] == 'FL']
        
        # SST2 comparison
        sst2_cent = centralized['Val_SST2_Acc'].mean()
        sst2_fl = fl['Val_SST2_Acc'].mean()
        
        if not pd.isna(sst2_cent) and not pd.isna(sst2_fl):
            comparison_data.append({
                'Model': model,
                'Task': 'SST2',
                'Centralized': sst2_cent,
                'FL': sst2_fl,
                'Total': sst2_cent + sst2_fl,
                'Difference': sst2_fl - sst2_cent,
                'Percent_Diff': ((sst2_fl - sst2_cent) / sst2_cent * 100)
            })
        
        # QQP comparison
        qqp_cent = centralized['Val_QQP_Acc'].mean()
        qqp_fl = fl['Val_QQP_Acc'].mean()
        
        if not pd.isna(qqp_cent) and not pd.isna(qqp_fl):
            comparison_data.append({
                'Model': model,
                'Task': 'QQP',
                'Centralized': qqp_cent,
                'FL': qqp_fl,
                'Total': qqp_cent + qqp_fl,
                'Difference': qqp_fl - qqp_cent,
                'Percent_Diff': ((qqp_fl - qqp_cent) / qqp_cent * 100)
            })
        
        # STSB comparison
        stsb_cent = centralized['Val_STSB_Pearson'].mean()
        stsb_fl = fl['Val_STSB_Pearson'].mean()
        
        if not pd.isna(stsb_cent) and not pd.isna(stsb_fl):
            comparison_data.append({
                'Model': model,
                'Task': 'STSB',
                'Centralized': stsb_cent,
                'FL': stsb_fl,
                'Total': stsb_cent + stsb_fl,
                'Difference': stsb_fl - stsb_cent,
                'Percent_Diff': ((stsb_fl - stsb_cent) / stsb_cent * 100)
            })
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        
        # Sort by total performance (descending)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(18, 10))
        
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, viz_df['Centralized'], label='Centralized', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, viz_df['FL'], bottom=viz_df['Centralized'], label='FL', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Centralized vs FL: Performance Comparison', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add balanced-enhanced value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Centralized value
            height1 = bar1.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                   f'{height1:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE, 
                   color='white', fontweight='bold')
            
            # FL value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + viz_df.iloc[i]['Centralized'],
                   f'{height2:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = viz_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + 0.01,
                   f'{total:.3f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE, 
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Performance comparison between Centralized and Federated Learning (FL) paradigms. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Performance Hierarchy**: Clear ranking of model-task combinations by total performance\n- **Paradigm Contribution**: Visual representation of each paradigm's contribution to total\n- **Top Performers**: Best configurations show strong performance from both paradigms\n- **Task Patterns**: Different tasks show different paradigm dominance patterns"
        
        metrics_df = viz_df[['Model', 'Task', 'Centralized', 'FL', 'Total', 'Difference', 'Percent_Diff']]
        save_plot_with_metadata(fig, 'centralized_vs_fl_stacked_performance_balanced',
                             'Centralized vs FL: Performance Comparison', description, insights, metrics_df)

def plot_single_vs_multitask_stacked_performance_balanced():
    """performance comparison: Single vs Multi-task"""
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        single_task = model_data[model_data['Task_Type'] == 'Single-Task']
        multi_task = model_data[model_data['Task_Type'] == 'Multi-Task']
        
        # SST2 comparison
        sst2_single = single_task['Val_SST2_Acc'].mean()
        sst2_multi = multi_task['Val_SST2_Acc'].mean()
        
        if not pd.isna(sst2_single) and not pd.isna(sst2_multi):
            comparison_data.append({
                'Model': model,
                'Task': 'SST2',
                'Single': sst2_single,
                'Multi': sst2_multi,
                'Total': sst2_single + sst2_multi,
                'Difference': sst2_multi - sst2_single,
                'Percent_Diff': ((sst2_multi - sst2_single) / sst2_single * 100)
            })
        
        # QQP comparison
        qqp_single = single_task['Val_QQP_Acc'].mean()
        qqp_multi = multi_task['Val_QQP_Acc'].mean()
        
        if not pd.isna(qqp_single) and not pd.isna(qqp_multi):
            comparison_data.append({
                'Model': model,
                'Task': 'QQP',
                'Single': qqp_single,
                'Multi': qqp_multi,
                'Total': qqp_single + qqp_multi,
                'Difference': qqp_multi - qqp_single,
                'Percent_Diff': ((qqp_multi - qqp_single) / qqp_single * 100)
            })
        
        # STSB comparison
        stsb_single = single_task['Val_STSB_Pearson'].mean()
        stsb_multi = multi_task['Val_STSB_Pearson'].mean()
        
        if not pd.isna(stsb_single) and not pd.isna(stsb_multi):
            comparison_data.append({
                'Model': model,
                'Task': 'STSB',
                'Single': stsb_single,
                'Multi': stsb_multi,
                'Total': stsb_single + stsb_multi,
                'Difference': stsb_multi - stsb_single,
                'Percent_Diff': ((stsb_multi - stsb_single) / stsb_single * 100)
            })
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        
        # Sort by total performance (descending)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(18, 10))
        
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, viz_df['Single'], label='Single-Task', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, viz_df['Multi'], bottom=viz_df['Single'], label='Multi-Task', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Single-Task vs Multi-Task: Performance Comparison', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add balanced-enhanced value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Single value
            height1 = bar1.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                   f'{height1:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Multi value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + viz_df.iloc[i]['Single'],
                   f'{height2:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = viz_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + 0.01,
                   f'{total:.3f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "performance comparison between Single-Task and Multi-Task Learning approaches. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Performance Ranking**: Clear hierarchy of model-task combinations by total performance\n- **Task Contribution**: Visual representation of each approach's contribution to total\n- **Transfer Effects**: Height differences show transfer learning benefits or interference\n- **Model Patterns**: Different models show different single vs multi-task balance"
        
        metrics_df = viz_df[['Model', 'Task', 'Single', 'Multi', 'Total', 'Difference', 'Percent_Diff']]
        save_plot_with_metadata(fig, 'single_vs_multitask_stacked_performance_balanced',
                             'Single vs Multi-Task: Performance Comparison', description, insights, metrics_df)

def plot_iid_vs_noniid_stacked_performance_balanced():
    """performance comparison: IID vs Non-IID"""
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        iid_data = model_data[model_data['Distribution'] == 'IID']
        non_iid_data = model_data[model_data['Distribution'] == 'Non-IID']
        
        if iid_data.empty or non_iid_data.empty:
            continue
        
        # SST2 comparison
        sst2_iid = iid_data['Val_SST2_Acc'].mean()
        sst2_non_iid = non_iid_data['Val_SST2_Acc'].mean()
        
        if not pd.isna(sst2_iid) and not pd.isna(sst2_non_iid):
            comparison_data.append({
                'Model': model,
                'Task': 'SST2',
                'IID': sst2_iid,
                'Non-IID': sst2_non_iid,
                'Total': sst2_iid + sst2_non_iid,
                'Degradation': sst2_iid - sst2_non_iid,
                'Percent_Degrad': ((sst2_iid - sst2_non_iid) / sst2_iid * 100)
            })
        
        # QQP comparison
        qqp_iid = iid_data['Val_QQP_Acc'].mean()
        qqp_non_iid = non_iid_data['Val_QQP_Acc'].mean()
        
        if not pd.isna(qqp_iid) and not pd.isna(qqp_non_iid):
            comparison_data.append({
                'Model': model,
                'Task': 'QQP',
                'IID': qqp_iid,
                'Non-IID': qqp_non_iid,
                'Total': qqp_iid + qqp_non_iid,
                'Degradation': qqp_iid - qqp_non_iid,
                'Percent_Degrad': ((qqp_iid - qqp_non_iid) / qqp_iid * 100)
            })
        
        # STSB comparison
        stsb_iid = iid_data['Val_STSB_Pearson'].mean()
        stsb_non_iid = non_iid_data['Val_STSB_Pearson'].mean()
        
        if not pd.isna(stsb_iid) and not pd.isna(stsb_non_iid):
            comparison_data.append({
                'Model': model,
                'Task': 'STSB',
                'IID': stsb_iid,
                'Non-IID': stsb_non_iid,
                'Total': stsb_iid + stsb_non_iid,
                'Degradation': stsb_iid - stsb_non_iid,
                'Percent_Degrad': ((stsb_iid - stsb_non_iid) / stsb_iid * 100)
            })
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        
        # Sort by total performance (descending)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(18, 10))
        
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, viz_df['IID'], label='IID', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, viz_df['Non-IID'], bottom=viz_df['IID'], label='Non-IID', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('IID vs Non-IID: Performance Comparison', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add balanced-enhanced value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # IID value
            height1 = bar1.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                   f'{height1:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Non-IID value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + viz_df.iloc[i]['IID'],
                   f'{height2:.3f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = viz_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + 0.01,
                   f'{total:.3f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Performance comparison between IID and Non-IID data distributions. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Performance Hierarchy**: Clear ranking of model-task combinations by total performance\n- **Distribution Impact**: Visual representation of each distribution's contribution to total\n- **Robustness Patterns**: Height ratios indicate model robustness to distribution shifts\n- **Task Sensitivity**: Different tasks show different IID vs Non-IID balance"
        
        metrics_df = viz_df[['Model', 'Task', 'IID', 'Non-IID', 'Total', 'Degradation', 'Percent_Degrad']]
        save_plot_with_metadata(fig, 'iid_vs_noniid_stacked_performance_balanced',
                             'IID vs Non-IID: Performance Comparison', description, insights, metrics_df)

def plot_fl_mtl_comparison_stacked_performance_balanced():
    """performance comparison: Centralized vs FL-Single vs FL-MTL"""
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        centralized = model_data[model_data['Paradigm'] == 'Centralized']
        fl_single = model_data[(model_data['Paradigm'] == 'FL') & (model_data['Task_Type'] == 'Single-Task')]
        fl_multi = model_data[(model_data['Paradigm'] == 'FL') & (model_data['Task_Type'] == 'Multi-Task')]
        
        for task_key, acc_col in [('SST2', 'Val_SST2_Acc'), ('QQP', 'Val_QQP_Acc'), ('STSB', 'Val_STSB_Pearson')]:
            c_val = centralized[acc_col].mean()
            fs_val = fl_single[acc_col].mean()
            fm_val = fl_multi[acc_col].mean()
            
            if not pd.isna(c_val) and not pd.isna(fs_val) and not pd.isna(fm_val):
                comparison_data.append({
                    'Model': model,
                    'Task': task_key,
                    'Centralized': c_val,
                    'FL_Single': fs_val,
                    'FL_Multi': fm_val,
                    'Total': c_val + fs_val + fm_val
                })
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Create three-way stacked bars
        bars1 = ax.bar(x_pos, viz_df['Centralized'], label='Centralized', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, viz_df['FL_Single'], bottom=viz_df['Centralized'], label='FL Single', alpha=0.8, color='#ff7f0e')
        bars3 = ax.bar(x_pos, viz_df['FL_Multi'], bottom=viz_df['Centralized'] + viz_df['FL_Single'], 
                       label='FL Multi-Task', alpha=0.8, color='#2ca02c')
        
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Three-Way Performance Analysis: Centralized vs FL Single vs FL-MTL', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE - 2, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
            for b, col in [(b1, 'Centralized'), (b2, 'FL_Single'), (b3, 'FL_Multi')]:
                h = b.get_height()
                y = b.get_y() + h/2
                ax.text(b.get_x() + b.get_width()/2., y, f'{h:.3f}', 
                        ha='center', va='center', fontsize=BALANCED_VALUE_SIZE - 2, 
                        color='white', fontweight='bold')
        
        description = "Three-way stacked performance comparison between Centralized baseline, FL Single-Task, and FL Multi-Task (MTL). This visualization highlights the performance gaps and gains when moving from single-task to multi-task learning in a federated environment."
        insights = "- **Baseline Comparison**: Centralized remains the upper bound for most model-task pairs.\n- **MTL Impact**: The third segment (Green) shows the contribution of Multi-Task Learning in FL settings.\n- **Model Sensitivity**: Different models respond differently to the transition from single to multi-task in FL."
        
        metrics_df = viz_df[['Model', 'Task', 'Centralized', 'FL_Single', 'FL_Multi', 'Total']]
        return save_plot_with_metadata(fig, "fl_mtl_comparison_stacked_performance_balanced", 
                                     "FL-MTL Performance Comparison Analysis", description, insights, metrics_df)

def plot_comprehensive_paradigm_task_stacked_performance_balanced():
    """performance comparison: Centralized-Single, Centralized-Multi, FL-Single, FL-Multi"""
    
    # Prepare data
    comparison_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        c_single = model_data[(model_data['Paradigm'] == 'Centralized') & (model_data['Task_Type'] == 'Single-Task')]
        c_multi = model_data[(model_data['Paradigm'] == 'Centralized') & (model_data['Task_Type'] == 'Multi-Task')]
        f_single = model_data[(model_data['Paradigm'] == 'FL') & (model_data['Task_Type'] == 'Single-Task')]
        f_multi = model_data[(model_data['Paradigm'] == 'FL') & (model_data['Task_Type'] == 'Multi-Task')]
        
        for task_key, acc_col in [('SST2', 'Val_SST2_Acc'), ('QQP', 'Val_QQP_Acc'), ('STSB', 'Val_STSB_Pearson')]:
            cs_val = c_single[acc_col].mean()
            cm_val = c_multi[acc_col].mean()
            fs_val = f_single[acc_col].mean()
            fm_val = f_multi[acc_col].mean()
            
            if not pd.isna(cs_val) and not pd.isna(cm_val) and not pd.isna(fs_val) and not pd.isna(fm_val):
                comparison_data.append({
                    'Model': model,
                    'Task': task_key,
                    'Cent_Single': cs_val,
                    'Cent_Multi': cm_val,
                    'FL_Single': fs_val,
                    'FL_Multi': fm_val,
                    'Total': cs_val + cm_val + fs_val + fm_val
                })
    
    if comparison_data:
        viz_df = pd.DataFrame(comparison_data)
        viz_df = viz_df.sort_values('Total', ascending=False)
        
        fig, ax = plt.subplots(figsize=(22, 12))
        x_pos = np.arange(len(viz_df))
        labels = [f"{row['Model']}-{row['Task']}" for _, row in viz_df.iterrows()]
        
        # Create four-way stacked bars
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c']
        bars1 = ax.bar(x_pos, viz_df['Cent_Single'], label='Cent Single', alpha=0.9, color=colors[0])
        bars2 = ax.bar(x_pos, viz_df['Cent_Multi'], bottom=viz_df['Cent_Single'], 
                       label='Cent Multi', alpha=0.9, color=colors[1])
        bars3 = ax.bar(x_pos, viz_df['FL_Single'], bottom=viz_df['Cent_Single'] + viz_df['Cent_Multi'], 
                       label='FL Single', alpha=0.9, color=colors[2])
        bars4 = ax.bar(x_pos, viz_df['FL_Multi'], bottom=viz_df['Cent_Single'] + viz_df['Cent_Multi'] + viz_df['FL_Single'], 
                       label='FL Multi', alpha=0.9, color=colors[3])
        
        ax.set_xlabel('Model-Task Combinations', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Performance', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Comprehensive 4-Way Analysis: Paradigm vs Task Type', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE - 4, fontweight='bold')
        add_model_banding(ax, viz_df)
        color_model_xticks(ax)
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add value labels
        for i, (b1, b2, b3, b4) in enumerate(zip(bars1, bars2, bars3, bars4)):
            for b_idx, b in enumerate([b1, b2, b3, b4]):
                h = b.get_height()
                y = b.get_y() + h/2
                ax.text(b.get_x() + b.get_width()/2., y, f'{h:.2f}', 
                        ha='center', va='center', fontsize=BALANCED_VALUE_SIZE - 4, 
                        color='black' if b_idx == 1 else 'white', fontweight='bold')
        
        description = "Ultimate 4-way stacked performance comparison: Centralized Single-Task vs Centralized Multi-Task vs FL Single-Task vs FL Multi-Task. This summarizes how both paradigms and task types interact across models."
        insights = "- **Paradigm Dominance**: Centralized (Blues) vs FL (Orange/Green) performance gaps are clearly visualized.\n- **Task Type Impact**: Compare light blue vs dark blue and green vs orange to see MTL effects in each paradigm.\n- **Overall Scaling**: Shows which models maintain performance best across all four rigorous conditions."
        
        metrics_df = viz_df[['Model', 'Task', 'Cent_Single', 'Cent_Multi', 'FL_Single', 'FL_Multi', 'Total']]
        return save_plot_with_metadata(fig, "comprehensive_paradigm_task_stacked_performance_balanced", 
                                     "Comprehensive Paradigm vs Task Performance Analysis", description, insights, metrics_df)
    return None, None

def generate_balanced_enhanced_stacked_plots():
    """Generate all comparison plots"""
    
    print("🔄 Generating Comparison Plots...")
    print("=" * 60)
    
    plots_generated = []
    
    # Generate each comparison
    balanced_functions = [
        plot_centralized_vs_fl_stacked_performance_balanced,
        plot_single_vs_multitask_stacked_performance_balanced,
        plot_iid_vs_noniid_stacked_performance_balanced,
        plot_fl_mtl_comparison_stacked_performance_balanced,
        plot_comprehensive_paradigm_task_stacked_performance_balanced
    ]
    
    for balanced_func in balanced_functions:
        try:
            balanced_func()
            plot_name = balanced_func.__name__.replace('plot_', '').replace('_balanced', '').replace('_', ' ').title()
            plots_generated.append(plot_name)
            print(f"✅ Generated: {plot_name}")
        except Exception as e:
            print(f"❌ Failed to generate {balanced_func.__name__}: {str(e)}")
    
    print(f"\n📊 Generated {len(plots_generated)} comparison plots")
    print(f"📁 All plots saved to: {plots_dir.absolute()}")
    print(f"📄 Markdown documentation with metrics included for each plot")
    print(f"⚖️ BALANCED font sizes (1.5x larger) for optimal readability")
    print(f"📦 Perfect balance between readability and space efficiency!")

if __name__ == "__main__":
    generate_balanced_enhanced_stacked_plots()
