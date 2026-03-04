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

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18   # was 12, now 1.5x
BALANCED_TITLE_SIZE = 21      # was 14, now 1.5x  
BALANCED_LEGEND_SIZE = 17    # was 11, now 1.5x
BALANCED_TICK_SIZE = 13       # was 8, now 1.5x
BALANCED_VALUE_SIZE = 13      # was 8, now 1.5x
BALANCED_XTICK_SIZE = 18      # was 12, now 1.5x for model names

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
        ax.set_title('Centralized vs FL: Performance Comparison (Sorted by Total)', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
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
        multi_task = model_data[model_data['Task_Type'] == 'Multi-Task (MTL)']
        
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
        ax.set_title('Single vs Multi-Task: Performance Comparison (Sorted by Total)', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
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
        ax.set_title('IID vs Non-IID: Performance Comparison (Sorted by Total)', 
                    fontsize=BALANCED_TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
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

def generate_balanced_enhanced_stacked_plots():
    """Generate all comparison plots"""
    
    print("🔄 Generating Comparison Plots...")
    print("=" * 60)
    
    plots_generated = []
    
    # Generate each comparison
    balanced_functions = [
        plot_centralized_vs_fl_stacked_performance_balanced,
        plot_single_vs_multitask_stacked_performance_balanced,
        plot_iid_vs_noniid_stacked_performance_balanced
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
