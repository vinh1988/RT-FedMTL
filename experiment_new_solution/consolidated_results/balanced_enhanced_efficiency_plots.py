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
BALANCED_TICK_SIZE = 12       # was 8, now 1.5x
BALANCED_VALUE_SIZE = 12      # was 8, now 1.5x
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

def plot_centralized_vs_fl_stacked_time_balanced():
    """Balanced-enhanced stacked training time comparison: Centralized vs FL"""
    
    # Prepare data
    time_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        centralized = model_data[model_data['Paradigm'] == 'Centralized']
        fl = model_data[model_data['Paradigm'] == 'FL']
        
        cent_time = centralized['Total_Train_Time'].mean()
        fl_time = fl['Total_Train_Time'].mean()
        
        if not pd.isna(cent_time) and not pd.isna(fl_time):
            time_data.append({
                'Model': model,
                'Centralized': cent_time,
                'FL': fl_time,
                'Total': cent_time + fl_time,
                'Ratio': fl_time / cent_time,
                'Difference': fl_time - cent_time
            })
    
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        # Sort by total training time (descending)
        time_df = time_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(time_df))
        labels = time_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, time_df['Centralized'], label='Centralized', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, time_df['FL'], bottom=time_df['Centralized'], label='FL', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Training Time (seconds)', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Centralized vs FL: Balanced-Enhanced Stacked Training Time Comparison (Sorted by Total)', 
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
                   f'{height1:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # FL value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + time_df.iloc[i]['Centralized'],
                   f'{height2:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = time_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.0f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked training time comparison between Centralized and Federated Learning (FL) paradigms. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Time Hierarchy**: Clear ranking of models by total training requirements\n- **FL Overhead**: Visual representation of FL's additional training time\n- **Model Scaling**: Larger models require more time for both paradigms\n- **Efficiency Patterns**: Different models show different time ratios"
        
        save_plot_with_metadata(fig, 'centralized_vs_fl_stacked_training_time_balanced',
                             'Centralized vs FL: Balanced-Enhanced Stacked Training Time Comparison', description, insights, time_df)

def plot_single_vs_multitask_stacked_time_balanced():
    """Balanced-enhanced stacked training time comparison: Single vs Multi-task"""
    
    # Prepare data
    time_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        single_task = model_data[model_data['Task_Type'] == 'Single-Task']
        multi_task = model_data[model_data['Task_Type'] == 'Multi-Task (MTL)']
        
        single_time = single_task['Total_Train_Time'].mean()
        multi_time = multi_task['Total_Train_Time'].mean()
        
        if not pd.isna(single_time) and not pd.isna(multi_time):
            time_data.append({
                'Model': model,
                'Single': single_time,
                'Multi': multi_time,
                'Total': single_time + multi_time,
                'Ratio': multi_time / single_time,
                'Difference': multi_time - single_time
            })
    
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        # Sort by total training time (descending)
        time_df = time_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(time_df))
        labels = time_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, time_df['Single'], label='Single-Task', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, time_df['Multi'], bottom=time_df['Single'], label='Multi-Task', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Training Time (seconds)', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Single vs Multi-Task: Balanced-Enhanced Stacked Training Time Comparison (Sorted by Total)', 
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
                   f'{height1:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Multi value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + time_df.iloc[i]['Single'],
                   f'{height2:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = time_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.0f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked training time comparison between Single-Task and Multi-Task Learning approaches. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Time Requirements**: Clear ranking of models by total training needs\n- **Multi-Task Efficiency**: Visual representation of multi-task training overhead\n- **Model Patterns**: Different models show different single vs multi-task time ratios\n- **Scalability Effects**: Training time scaling patterns across model sizes"
        
        save_plot_with_metadata(fig, 'single_vs_multitask_stacked_training_time_balanced',
                             'Single vs Multi-Task: Balanced-Enhanced Stacked Training Time Comparison', description, insights, time_df)

def plot_iid_vs_noniid_stacked_time_balanced():
    """Balanced-enhanced stacked training time comparison: IID vs Non-IID"""
    
    # Prepare data
    time_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        iid_data = model_data[model_data['Distribution'] == 'IID']
        non_iid_data = model_data[model_data['Distribution'] == 'Non-IID']
        
        if iid_data.empty or non_iid_data.empty:
            continue
        
        iid_time = iid_data['Total_Train_Time'].mean()
        non_iid_time = non_iid_data['Total_Train_Time'].mean()
        
        if not pd.isna(iid_time) and not pd.isna(non_iid_time):
            time_data.append({
                'Model': model,
                'IID': iid_time,
                'Non-IID': non_iid_time,
                'Total': iid_time + non_iid_time,
                'Ratio': non_iid_time / iid_time,
                'Difference': non_iid_time - iid_time
            })
    
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        # Sort by total training time (descending)
        time_df = time_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(time_df))
        labels = time_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, time_df['IID'], label='IID', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, time_df['Non-IID'], bottom=time_df['IID'], label='Non-IID', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Training Time (seconds)', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('IID vs Non-IID: Balanced-Enhanced Stacked Training Time Comparison (Sorted by Total)', 
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
                   f'{height1:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Non-IID value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + time_df.iloc[i]['IID'],
                   f'{height2:.0f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = time_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.0f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked training time comparison between IID and Non-IID data distributions. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Convergence Patterns**: Different models show different convergence requirements\n- **Distribution Impact**: Visual representation of Non-IID training overhead\n- **Model Adaptation**: Training time scaling with distribution complexity\n- **Efficiency Patterns**: Some models handle Non-IID more efficiently"
        
        save_plot_with_metadata(fig, 'iid_vs_noniid_stacked_training_time_balanced',
                             'IID vs Non-IID: Balanced-Enhanced Stacked Training Time Comparison', description, insights, time_df)

def plot_centralized_vs_fl_stacked_resource_balanced():
    """Balanced-enhanced stacked resource usage comparison: Centralized vs FL"""
    
    # Prepare data
    resource_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        centralized = model_data[model_data['Paradigm'] == 'Centralized']
        fl = model_data[model_data['Paradigm'] == 'FL']
        
        cent_resource = centralized['Resource_Usage'].mean()
        fl_resource = fl['Resource_Usage'].mean()
        
        if not pd.isna(cent_resource) and not pd.isna(fl_resource):
            resource_data.append({
                'Model': model,
                'Centralized': cent_resource,
                'FL': fl_resource,
                'Total': cent_resource + fl_resource,
                'Ratio': fl_resource / cent_resource,
                'Difference': fl_resource - cent_resource
            })
    
    if resource_data:
        resource_df = pd.DataFrame(resource_data)
        
        # Sort by total resource usage (descending)
        resource_df = resource_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(resource_df))
        labels = resource_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, resource_df['Centralized'], label='Centralized', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, resource_df['FL'], bottom=resource_df['Centralized'], label='FL', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Resource Usage', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Centralized vs FL: Balanced-Enhanced Stacked Resource Usage Comparison (Sorted by Total)', 
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
                   f'{height1:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # FL value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + resource_df.iloc[i]['Centralized'],
                   f'{height2:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = resource_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.2f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked resource usage comparison between Centralized and Federated Learning (FL) paradigms. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Resource Hierarchy**: Clear ranking of models by total resource requirements\n- **FL Efficiency**: Visual representation of FL's distributed resource efficiency\n- **Model Scaling**: Resource usage patterns across different model sizes\n- **Deployment Patterns**: Different resource requirements for each paradigm"
        
        save_plot_with_metadata(fig, 'centralized_vs_fl_stacked_resource_usage_balanced',
                             'Centralized vs FL: Balanced-Enhanced Stacked Resource Usage Comparison', description, insights, resource_df)

def plot_single_vs_multitask_stacked_resource_balanced():
    """Balanced-enhanced stacked resource usage comparison: Single vs Multi-task"""
    
    # Prepare data
    resource_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        single_task = model_data[model_data['Task_Type'] == 'Single-Task']
        multi_task = model_data[model_data['Task_Type'] == 'Multi-Task (MTL)']
        
        single_resource = single_task['Resource_Usage'].mean()
        multi_resource = multi_task['Resource_Usage'].mean()
        
        if not pd.isna(single_resource) and not pd.isna(multi_resource):
            resource_data.append({
                'Model': model,
                'Single': single_resource,
                'Multi': multi_resource,
                'Total': single_resource + multi_resource,
                'Ratio': multi_resource / single_resource,
                'Difference': multi_resource - single_resource
            })
    
    if resource_data:
        resource_df = pd.DataFrame(resource_data)
        
        # Sort by total resource usage (descending)
        resource_df = resource_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(resource_df))
        labels = resource_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, resource_df['Single'], label='Single-Task', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, resource_df['Multi'], bottom=resource_df['Single'], label='Multi-Task', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Resource Usage', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('Single vs Multi-Task: Balanced-Enhanced Stacked Resource Usage Comparison (Sorted by Total)', 
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
                   f'{height1:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Multi value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + resource_df.iloc[i]['Single'],
                   f'{height2:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = resource_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.2f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked resource usage comparison between Single-Task and Multi-Task Learning approaches. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Resource Efficiency**: Clear ranking of models by total resource consumption\n- **Multi-Task Benefits**: Visual representation of shared parameter efficiency\n- **Model Patterns**: Different models show different resource scaling\n- **Deployment Considerations**: Resource requirements for different task configurations"
        
        save_plot_with_metadata(fig, 'single_vs_multitask_stacked_resource_usage_balanced',
                             'Single vs Multi-Task: Balanced-Enhanced Stacked Resource Usage Comparison', description, insights, resource_df)

def plot_iid_vs_noniid_stacked_resource_balanced():
    """Balanced-enhanced stacked resource usage comparison: IID vs Non-IID"""
    
    # Prepare data
    resource_data = []
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        iid_data = model_data[model_data['Distribution'] == 'IID']
        non_iid_data = model_data[model_data['Distribution'] == 'Non-IID']
        
        if iid_data.empty or non_iid_data.empty:
            continue
        
        iid_resource = iid_data['Resource_Usage'].mean()
        non_iid_resource = non_iid_data['Resource_Usage'].mean()
        
        if not pd.isna(iid_resource) and not pd.isna(non_iid_resource):
            resource_data.append({
                'Model': model,
                'IID': iid_resource,
                'Non-IID': non_iid_resource,
                'Total': iid_resource + non_iid_resource,
                'Ratio': non_iid_resource / iid_resource,
                'Difference': non_iid_resource - iid_resource
            })
    
    if resource_data:
        resource_df = pd.DataFrame(resource_data)
        
        # Sort by total resource usage (descending)
        resource_df = resource_df.sort_values('Total', ascending=False)
        
        # Create stacked bar plot with balanced-enhanced fonts
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x_pos = np.arange(len(resource_df))
        labels = resource_df['Model'].tolist()
        
        # Create stacked bars
        bars1 = ax.bar(x_pos, resource_df['IID'], label='IID', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x_pos, resource_df['Non-IID'], bottom=resource_df['IID'], label='Non-IID', alpha=0.8, color='#ff7f0e')
        
        # Balanced-enhanced labels and fonts
        ax.set_xlabel('Model', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_ylabel('Stacked Resource Usage', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')
        ax.set_title('IID vs Non-IID: Balanced-Enhanced Stacked Resource Usage Comparison (Sorted by Total)', 
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
                   f'{height1:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Non-IID value
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2/2 + resource_df.iloc[i]['IID'],
                   f'{height2:.2f}', ha='center', va='center', fontsize=BALANCED_VALUE_SIZE,
                   color='white', fontweight='bold')
            
            # Total value on top
            total = resource_df.iloc[i]['Total']
            ax.text(bar2.get_x() + bar2.get_width()/2., total + total*0.02,
                   f'{total:.2f}', ha='center', va='bottom', fontsize=BALANCED_VALUE_SIZE,
                   fontweight='bold', color='darkred')
        
        plt.tight_layout()
        
        description = "Balanced-enhanced stacked resource usage comparison between IID and Non-IID data distributions. All text and numbers are 1.5x larger for optimal readability."
        insights = "- **Resource Scaling**: Different models show different resource requirements\n- **Distribution Impact**: Visual representation of Non-IID resource efficiency\n- **Model Adaptation**: Resource usage patterns with distribution complexity\n- **Efficiency Patterns**: Some models handle Non-IID more resource-efficiently"
        
        save_plot_with_metadata(fig, 'iid_vs_noniid_stacked_resource_usage_balanced',
                             'IID vs Non-IID: Balanced-Enhanced Stacked Resource Usage Comparison', description, insights, resource_df)

def generate_balanced_enhanced_efficiency_plots():
    """Generate all balanced-enhanced stacked efficiency comparison plots"""
    
    print("🔄 Generating Balanced-Enhanced Stacked Efficiency Comparison Plots...")
    print("=" * 60)
    
    plots_generated = []
    
    # Generate each balanced-enhanced stacked comparison
    balanced_functions = [
        plot_centralized_vs_fl_stacked_time_balanced,
        plot_single_vs_multitask_stacked_time_balanced,
        plot_iid_vs_noniid_stacked_time_balanced,
        plot_centralized_vs_fl_stacked_resource_balanced,
        plot_single_vs_multitask_stacked_resource_balanced,
        plot_iid_vs_noniid_stacked_resource_balanced
    ]
    
    for balanced_func in balanced_functions:
        try:
            balanced_func()
            plot_name = balanced_func.__name__.replace('plot_', '').replace('_balanced', '').replace('_', ' ').title()
            plots_generated.append(plot_name)
            print(f"✅ Generated: {plot_name}")
        except Exception as e:
            print(f"❌ Failed to generate {balanced_func.__name__}: {str(e)}")
    
    print(f"\n📊 Generated {len(plots_generated)} balanced-enhanced stacked efficiency comparison plots")
    print(f"📁 All plots saved to: {plots_dir.absolute()}")
    print(f"📄 Markdown documentation with metrics included for each plot")
    print(f"⚖️ BALANCED font sizes (1.5x larger) for optimal readability")
    print(f"📦 Perfect balance between readability and space efficiency!")

if __name__ == "__main__":
    generate_balanced_enhanced_efficiency_plots()
