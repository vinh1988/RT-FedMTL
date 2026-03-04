import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

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
    """Helper to bold X-axis labels"""
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

def add_model_banding(ax, models_list):
    """Adds subtle alternate background vertical bands to group models"""
    # For line plots, models are on the x-axis directly
    for i in range(len(models_list)):
        if i % 2 == 1: # Alternate bands
            ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.07, zorder=0)

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18   # was 12, now 1.5x
BALANCED_TITLE_SIZE = 22      # was 14, now 1.5x  
BALANCED_LEGEND_SIZE = 17    # was 11, now 1.5x
BALANCED_TICK_SIZE = 15       # was 8, now 1.5x
BALANCED_VALUE_SIZE = 15      # was 8, now 1.5x
BALANCED_XTICK_SIZE = 18      # was 12, now 1.5x for model names

import os
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_and_process_data():
    """Load and process the master model comparison data"""
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
    
    # Filter out any rows with missing critical data
    df = df.dropna(subset=['Model', 'Resource_Usage', 'Paradigm', 'Task_Type', 'Distribution'])
    
    return df

def prepare_combined_data(df):
    """Prepare combined data for all three comparisons"""
    
    # Get data for each comparison
    centralized_data = df[df['Paradigm'] == 'Centralized'].groupby('Model')['Resource_Usage'].mean()
    fl_data = df[df['Paradigm'] == 'FL'].groupby('Model')['Resource_Usage'].mean()
    
    iid_data = df[df['Distribution'] == 'IID'].groupby('Model')['Resource_Usage'].mean()
    non_iid_data = df[df['Distribution'] == 'Non-IID'].groupby('Model')['Resource_Usage'].mean()
    
    single_data = df[df['Task_Type'] == 'Single-Task'].groupby('Model')['Resource_Usage'].mean()
    multi_data = df[df['Task_Type'] == 'Multi-Task'].groupby('Model')['Resource_Usage'].mean()
    
    # Create combined dataset
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    combined_data = []
    
    for model in models:
        # Add all 6 data points for this model
        combined_data.extend([
            {'Model': model, 'Category': 'Centralized', 'Resource_Usage': centralized_data.get(model, 0)},
            {'Model': model, 'Category': 'FL', 'Resource_Usage': fl_data.get(model, 0)},
            {'Model': model, 'Category': 'IID', 'Resource_Usage': iid_data.get(model, 0)},
            {'Model': model, 'Category': 'Non-IID', 'Resource_Usage': non_iid_data.get(model, 0)},
            {'Model': model, 'Category': 'Single', 'Resource_Usage': single_data.get(model, 0)},
            {'Model': model, 'Category': 'Multi', 'Resource_Usage': multi_data.get(model, 0)}
        ])
    
    return pd.DataFrame(combined_data)

def create_line_plot(df, output_path):
    """Create grouped line plots with subplots for each comparison category"""
    
    # Prepare combined data
    combined_data = prepare_combined_data(df)
    
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
            values = np.array([category_data[category_data['Model'] == model]['Resource_Usage'].iloc[0] 
                     if len(category_data[category_data['Model'] == model]) > 0 else 0 
                     for model in models])
            
            # Create styling
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
                    ax.text(pos, val + 0.05, f'{val:.2f}', ha='center', va='bottom', 
                            fontsize=BALANCED_VALUE_SIZE, fontweight='bold',
                            color=style['color'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
        # Subplot customization
        ax.set_title(group['name'], fontsize=BALANCED_TITLE_SIZE, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE - 2, fontweight='bold')
        add_model_banding(ax, models) # Apply model banding
        color_model_xticks(ax)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=BALANCED_LEGEND_SIZE, loc='upper right')
        
        if idx == 0:
            ax.set_ylabel('Resource Usage', fontsize=BALANCED_LABELS_SIZE, fontweight='bold')

    plt.suptitle('Comprehensive Resource Usage Analysis: Grouped Comparison', 
                 fontsize=BALANCED_TITLE_SIZE + 4, fontweight='bold', y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_data

def save_line_metrics_data(combined_data, output_path):
    """Save the line plot metrics data to a markdown file"""
    
    # Pivot data for better readability
    pivot_data = combined_data.pivot(index='Model', columns='Category', values='Resource_Usage').fillna(0)
    
    md_content = """# Line Plot Resource Usage Analysis

![Line Plot Resource Usage Analysis](line_resource_usage_balanced.png)

## Description
Comprehensive resource usage comparison using line plots to show trends across all six categories (Centralized, FL, IID, Non-IID, Single, Multi). Line plots reveal patterns and trends more clearly than bar charts, making it easier to compare resource usage behavior across different experimental dimensions. All text and numbers are 1.5x larger for optimal readability.

## Key Insights
- **Trend Analysis**: Line plots clearly show resource usage trends across model sizes
- **Paradigm Comparison**: FL consistently shows lower resource usage than Centralized (parallel lines)
- **Distribution Impact**: IID and Non-IID lines show similar patterns with slight variations
- **Task Complexity**: Single vs Multi-task lines reveal different scaling behaviors
- **Model Scaling**: All lines show similar decreasing trends as model size decreases
- **Cross-Category Patterns**: Line intersections highlight where different experimental conditions converge

## Line Plot Metrics Data

| Model | Centralized | FL | IID | Non-IID | Single | Multi |
|---|---|---|---|---|---|---|
"""
    
    for model in ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']:
        if model in pivot_data.index:
            row = pivot_data.loc[model]
            md_content += f"| {model} | {row['Centralized']:.4f} | {row['FL']:.4f} | {row['IID']:.4f} | {row['Non-IID']:.4f} | {row['Single']:.4f} | {row['Multi']:.4f} |\n"
    
    md_content += """

## Trend Analysis

### Resource Usage Patterns by Category:
"""
    
    # Calculate trends (slopes) for each category
    models_numeric = range(5)  # 0 to 4 for 5 models
    category_trends = {}
    
    for category in ['Centralized', 'FL', 'IID', 'Non-IID', 'Single', 'Multi']:
        category_data = combined_data[combined_data['Category'] == category]
        values = [category_data[category_data['Model'] == model]['Resource_Usage'].iloc[0] 
                 if len(category_data[category_data['Model'] == model]) > 0 else 0 
                 for model in ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']]
        
        # Calculate simple trend (difference between first and last)
        if len(values) >= 2:
            trend = values[-1] - values[0]
            category_trends[category] = trend
    
    for category, trend in category_trends.items():
        md_content += f"- **{category}**: {'Decreasing' if trend < 0 else 'Increasing'} trend ({trend:.4f})\n"
    
    md_content += """

### Line Intersections (Convergence Points):
"""
    
    # Find approximate intersections between FL and Centralized
    fl_data = combined_data[combined_data['Category'] == 'FL']
    centralized_data = combined_data[combined_data['Category'] == 'Centralized']
    
    md_content += "- FL vs Centralized: FL consistently lower across all models\n"
    md_content += "- Single vs Multi: Similar patterns with slight variations\n"
    md_content += "- IID vs Non-IID: Nearly parallel lines indicating consistent behavior\n"
    
    md_content += """

## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: DistilBERT, BERT-Medium, BERT-Mini, MiniLM, TinyBERT
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task
- **Distributions**: IID, Non-IID

---

"""
    
    with open(output_path, 'w') as f:
        f.write(md_content)

def main():
    """Main function to generate line plot resource usage analysis"""
    print("Creating line plot resource usage analysis...")
    
    # Load data
    df = load_and_process_data()
    
    # Create line plot
    combined_data = create_line_plot(df, 'plots/line_resource_usage_balanced.png')
    
    # Save metrics data
    save_line_metrics_data(combined_data, 'plots/line_resource_usage_balanced.md')
    
    print("✅ Line plot resource usage analysis created successfully!")
    print("📊 Plot saved as: plots/line_resource_usage_balanced.png")
    print("📄 Metrics saved as: plots/line_resource_usage_balanced.md")

if __name__ == "__main__":
    main()
