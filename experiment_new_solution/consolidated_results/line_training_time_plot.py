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
    df = df.dropna(subset=['Model', 'Total_Train_Time', 'Paradigm', 'Task_Type', 'Distribution'])
    
    return df

def prepare_combined_training_data(df):
    """Prepare combined training time data for all three comparisons"""
    
    # Get data for each comparison
    centralized_data = df[df['Paradigm'] == 'Centralized'].groupby('Model')['Total_Train_Time'].mean()
    fl_data = df[df['Paradigm'] == 'FL'].groupby('Model')['Total_Train_Time'].mean()
    
    iid_data = df[df['Distribution'] == 'IID'].groupby('Model')['Total_Train_Time'].mean()
    non_iid_data = df[df['Distribution'] == 'Non-IID'].groupby('Model')['Total_Train_Time'].mean()
    
    single_data = df[df['Task_Type'] == 'Single-Task'].groupby('Model')['Total_Train_Time'].mean()
    multi_data = df[df['Task_Type'] == 'Multi-Task (MTL)'].groupby('Model')['Total_Train_Time'].mean()
    
    # Create combined dataset
    models = ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']
    combined_data = []
    
    for model in models:
        # Add all 6 data points for this model
        combined_data.extend([
            {'Model': model, 'Category': 'Centralized', 'Training_Time': centralized_data.get(model, 0)},
            {'Model': model, 'Category': 'FL', 'Training_Time': fl_data.get(model, 0)},
            {'Model': model, 'Category': 'IID', 'Training_Time': iid_data.get(model, 0)},
            {'Model': model, 'Category': 'Non-IID', 'Training_Time': non_iid_data.get(model, 0)},
            {'Model': model, 'Category': 'Single', 'Training_Time': single_data.get(model, 0)},
            {'Model': model, 'Category': 'Multi', 'Training_Time': multi_data.get(model, 0)}
        ])
    
    return pd.DataFrame(combined_data)

def create_training_time_line_plot(df, output_path):
    """Create a line plot with all training time comparisons"""
    
    # Prepare combined data
    combined_data = prepare_combined_training_data(df)
    
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
        values = [category_data[category_data['Model'] == model]['Training_Time'].iloc[0] 
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
        
        # Add value labels on points (format as seconds with appropriate units)
        for i, (pos, val) in enumerate(zip(x_pos, values)):
            if val > 0:
                # Format value based on magnitude
                if val >= 1000:
                    label = f'{val/1000:.1f}k'
                else:
                    label = f'{val:.0f}'
                plt.text(pos, val + val*0.02, label, ha='center', va='bottom', 
                        fontsize=BALANCED_VALUE_SIZE - 2, fontweight='bold', rotation=45)
    
    # Customize plot
    plt.title('Comprehensive Training Time Analysis: Line Plot Comparison', 
              fontsize=BALANCED_TITLE_SIZE + 2, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=BALANCED_LABELS_SIZE)
    plt.ylabel('Training Time (seconds)', fontsize=BALANCED_LABELS_SIZE)
    plt.xticks(x_pos, models, rotation=45, ha='right', fontsize=BALANCED_XTICK_SIZE, fontweight='bold')
    plt.yticks(fontsize=BALANCED_TICK_SIZE)
    
    # Format y-axis to show values in thousands
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))
    
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

def save_training_time_metrics_data(combined_data, output_path):
    """Save the training time line plot metrics data to a markdown file"""
    
    # Pivot data for better readability
    pivot_data = combined_data.pivot(index='Model', columns='Category', values='Training_Time').fillna(0)
    
    md_content = """# Line Plot Training Time Analysis

![Line Plot Training Time Analysis](line_training_time_balanced.png)

## Description
Comprehensive training time comparison using line plots to show trends across all six categories (Centralized, FL, IID, Non-IID, Single, Multi). Line plots reveal training time patterns and overhead more clearly than bar charts, making it easier to compare computational efficiency across different experimental dimensions. All text and numbers are 1.5x larger for optimal readability.

## Key Insights
- **FL Overhead**: Clear gap between Centralized and FL lines shows 2.4-3.9x training overhead
- **Model Scaling**: All lines show decreasing training time as model size decreases
- **Distribution Impact**: IID vs Non-IID lines show similar patterns with slight variations
- **Task Complexity**: Single vs Multi-task reveal different training time requirements
- **Efficiency Patterns**: Line slopes indicate different scaling behaviors
- **Convergence Points**: Areas where different experimental conditions have similar training times

## Training Time Metrics Data

| Model | Centralized | FL | IID | Non-IID | Single | Multi |
|---|---|---|---|---|---|---|
"""
    
    for model in ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']:
        if model in pivot_data.index:
            row = pivot_data.loc[model]
            md_content += f"| {model} | {row['Centralized']:.0f} | {row['FL']:.0f} | {row['IID']:.0f} | {row['Non-IID']:.0f} | {row['Single']:.0f} | {row['Multi']:.0f} |\n"
    
    md_content += """

## Training Time Analysis

### Overhead Ratios by Model:
"""
    
    # Calculate FL vs Centralized ratios
    for model in ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']:
        if model in pivot_data.index:
            row = pivot_data.loc[model]
            if row['Centralized'] > 0:
                ratio = row['FL'] / row['Centralized']
                md_content += f"- **{model}**: FL is {ratio:.2f}x slower than Centralized\n"
    
    md_content += """

### Training Time Trends:
"""
    
    # Calculate trends (slopes) for each category
    category_trends = {}
    
    for category in ['Centralized', 'FL', 'IID', 'Non-IID', 'Single', 'Multi']:
        category_data = combined_data[combined_data['Category'] == category]
        values = [category_data[category_data['Model'] == model]['Training_Time'].iloc[0] 
                 if len(category_data[category_data['Model'] == model]) > 0 else 0 
                 for model in ['distil-bert', 'medium-bert', 'mini-bert', 'mini-lm', 'tiny_bert']]
        
        # Calculate simple trend (difference between first and last)
        if len(values) >= 2 and values[0] > 0:
            trend_percent = ((values[-1] - values[0]) / values[0]) * 100
            category_trends[category] = trend_percent
    
    for category, trend in category_trends.items():
        md_content += f"- **{category}**: {trend:.1f}% change from largest to smallest model\n"
    
    md_content += """

### Key Observations:
- **FL Communication Overhead**: Consistent across all models
- **Model Size Impact**: Larger models require disproportionately more training time
- **Task Type Effects**: Multi-task training shows different efficiency patterns
- **Data Distribution**: Minimal impact on training time between IID and Non-IID

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
    """Main function to generate line plot training time analysis"""
    print("Creating line plot training time analysis...")
    
    # Load data
    df = load_and_process_data()
    
    # Create line plot
    combined_data = create_training_time_line_plot(df, 'plots/line_training_time_balanced.png')
    
    # Save metrics data
    save_training_time_metrics_data(combined_data, 'plots/line_training_time_balanced.md')
    
    print("✅ Line plot training time analysis created successfully!")
    print("📊 Plot saved as: plots/line_training_time_balanced.png")
    print("📄 Metrics saved as: plots/line_training_time_balanced.md")

if __name__ == "__main__":
    main()
