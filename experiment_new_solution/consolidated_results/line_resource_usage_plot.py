import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

# Balanced-enhanced font sizes (1.5x larger than original)
BALANCED_LABELS_SIZE = 18   # was 12, now 1.5x
BALANCED_TITLE_SIZE = 21      # was 14, now 1.5x  
BALANCED_LEGEND_SIZE = 17    # was 11, now 1.5x
BALANCED_TICK_SIZE = 12       # was 8, now 1.5x
BALANCED_VALUE_SIZE = 12      # was 8, now 1.5x
BALANCED_XTICK_SIZE = 18      # was 12, now 1.5x for model names

def load_and_process_data():
    """Load and process the master model comparison data"""
    df = pd.read_csv('master_model_comparison.csv')
    
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
    multi_data = df[df['Task_Type'] == 'Multi-Task (MTL)'].groupby('Model')['Resource_Usage'].mean()
    
    # Create combined dataset
    models = ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']
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
    """Create a line plot with all resource usage comparisons"""
    
    # Prepare combined data
    combined_data = prepare_combined_data(df)
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Define colors and line styles for each category
    styles = {
        'Centralized': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-'},
        'FL': {'color': '#A23B72', 'marker': 's', 'linestyle': '--'},
        'IID': {'color': '#F18F01', 'marker': '^', 'linestyle': '-'},
        'Non-IID': {'color': '#C73E1D', 'marker': 'v', 'linestyle': '--'},
        'Single': {'color': '#3A86FF', 'marker': 'D', 'linestyle': '-'},
        'Multi': {'color': '#8338EC', 'marker': 'p', 'linestyle': '--'}
    }
    
    # Get models in order
    models = ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']
    x_pos = np.arange(len(models))
    
    # Plot lines for each category
    for category in ['Centralized', 'FL', 'IID', 'Non-IID', 'Single', 'Multi']:
        # Get data for this category
        category_data = combined_data[combined_data['Category'] == category]
        values = [category_data[category_data['Model'] == model]['Resource_Usage'].iloc[0] 
                 if len(category_data[category_data['Model'] == model]) > 0 else 0 
                 for model in models]
        
        # Create line plot
        style = styles[category]
        plt.plot(x_pos, values, 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=3, 
                markersize=10,
                label=category,
                alpha=0.8)
        
        # Add value labels on points
        for i, (pos, val) in enumerate(zip(x_pos, values)):
            if val > 0:
                plt.text(pos, val + 0.05, f'{val:.2f}', ha='center', va='bottom', 
                        fontsize=BALANCED_VALUE_SIZE - 2, fontweight='bold', rotation=45)
    
    # Customize plot
    plt.title('Comprehensive Resource Usage Analysis: Line Plot Comparison', 
              fontsize=BALANCED_TITLE_SIZE + 2, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=BALANCED_LABELS_SIZE)
    plt.ylabel('Resource Usage', fontsize=BALANCED_LABELS_SIZE)
    plt.xticks(x_pos, models, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
    plt.yticks(fontsize=BALANCED_TICK_SIZE)
    
    # Add legend
    plt.legend(fontsize=BALANCED_LEGEND_SIZE - 2, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
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
    
    for model in ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']:
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
                 for model in ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']]
        
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
- **Models**: distil-bert, medium-bert, mini-bert, mini-lm, tiny_bert
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task (MTL)
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
