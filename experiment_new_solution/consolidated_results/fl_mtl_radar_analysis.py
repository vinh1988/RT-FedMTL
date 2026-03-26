import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Radar Chart Axis Labels
CATEGORIES = [
    'Peak Performance', 
    'Avg. Accuracy', 
    'Time Efficiency', 
    'Resource Efficiency', 
    'Versatility'
]

def apply_publication_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })

def get_global_radar_data(df):
    """Extract and normalize 5-axis radar data globally across all models"""
    perf_cols = ['Val_SST2_Acc', 'Val_QQP_Acc', 'Val_STSB_Pearson']
    
    # Pre-calculate global min/max for normalization
    global_time_min = df['Total_Train_Time'].min()
    global_time_max = df['Total_Train_Time'].max()
    global_res_min = df['Resource_Usage'].min()
    global_res_max = df['Resource_Usage'].max()
    
    radar_entries = []
    
    models = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
    
    for model in models:
        m_df = df[df['Model'] == model].copy()
        if m_df.empty: continue
        
        m_df['Peak_Perf'] = m_df[perf_cols].max(axis=1)
        m_df['Avg_Perf'] = m_df[perf_cols].mean(axis=1)
        
        # Centralized Best
        cent_best = m_df[m_df['Paradigm'] == 'Centralized'].sort_values('Avg_Perf', ascending=False).iloc[0]
        
        # FL-MTL Main (Proposed)
        fl_mtl_main = m_df[(m_df['Paradigm'] == 'FL') & (m_df['Experiment'].str.contains('lora', case=False))].sort_values('Avg_Perf', ascending=False)
        
        if not fl_mtl_main.empty:
            fl_mtl_main = fl_mtl_main.iloc[0]
            
            for row, label_suffix in [(cent_best, 'Centralized'), (fl_mtl_main, 'FL-MTL')]:
                # 5-Axis Calculation (0.0 to 1.0)
                p1 = row['Peak_Perf']
                p2 = row['Avg_Perf']
                # Time Efficiency (Inverted normalized)
                p3 = 1.0 - (row['Total_Train_Time'] - global_time_min) / (global_time_max - global_time_min)
                # Resource Efficiency (Inverted normalized)
                p4 = 1.0 - (row['Resource_Usage'] - global_res_min) / (global_res_max - global_res_min)
                # Versatility
                p5 = sum([1 for col in perf_cols if row[col] > 0.5]) / 3.0
                
                # Scale for visibility
                radar_vals = [0.2 + (v * 0.8) for v in [p1, p2, p3, p4, p5]]
                
                radar_entries.append({
                    'Model': model,
                    'Method': label_suffix,
                    'Values': radar_vals,
                    'Avg_Perf': row['Avg_Perf'],
                    'Raw_Time': row['Total_Train_Time'],
                    'Raw_Res': row['Resource_Usage']
                })
                
    return radar_entries

def draw_global_radar(radar_data):
    """Draw a single consolidated Radar chart with recommendations"""
    apply_publication_style()
    
    N = len(CATEGORIES)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Find Best Options
    best_cent = max([d for d in radar_data if d['Method'] == 'Centralized'], key=lambda x: x['Avg_Perf'])
    best_fl = max([d for d in radar_data if d['Method'] == 'FL-MTL'], key=lambda x: x['Avg_Perf'])
    
    plt.xticks(angles[:-1], CATEGORIES, fontweight='bold', size=15)
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8, 1.0], ["", "", "", ""], color="grey", size=10)
    plt.ylim(0, 1.1)

    for entry in radar_data:
        label = f"{entry['Model']} ({entry['Method']})"
        values = entry['Values']
        vals = values + values[:1]
        
        is_best_cent = (entry == best_cent)
        is_best_fl = (entry == best_fl)
        
        if is_best_cent:
            color = '#2c3e50'
            linewidth = 5
            linestyle = '-'
            label = f"⭐ RECOMMEND: Best Centralized ({entry['Model']})"
            alpha = 0.2
        elif is_best_fl:
            color = '#e74c3c'
            linewidth = 5
            linestyle = '-'
            label = f"🚀 RECOMMEND: Best FL-MTL ({entry['Model']})"
            alpha = 0.25
        else:
            color = '#7f8c8d' if entry['Method'] == 'Centralized' else '#3498db'
            linewidth = 1.5
            linestyle = '--' if entry['Method'] == 'Centralized' else ':'
            alpha = 0.05
            
        ax.plot(angles, vals, linewidth=linewidth, linestyle=linestyle, label=label, color=color, zorder=10 if (is_best_cent or is_best_fl) else 1)
        if is_best_cent or is_best_fl:
            ax.fill(angles, vals, color=color, alpha=alpha)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True, title="Methodology & Recommendations")
    plt.title('Global Performance-Efficiency Radar: Centralized vs. Decentralized', size=24, y=1.1, fontweight='bold')
    
    filename = "global_method_tradeoff_radar.png"
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {filename}")
    return best_cent, best_fl

def export_markdown_table(radar_data, best_cent, best_fl):
    """Save normalized radar data to a markdown table"""
    md_content = "# Consolidated Global Trade-off Data\n\n"
    md_content += "This table shows the normalized values (0.2 - 1.0 scale) used for the Radar chart analysis.\n\n"
    
    headers = ["Model", "Paradigm", "Peak Perf", "Avg Acc", "Time Eff", "Res Eff", "Versatility", "Status"]
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for d in sorted(radar_data, key=lambda x: (x['Method'], -x['Avg_Perf'])):
        vals = d['Values']
        status = ""
        if d == best_cent: status = "**Best Centralized**"
        if d == best_fl: status = "**Best FL-MTL**"
        
        row = [
            d['Model'], d['Method'],
            f"{vals[0]:.3f}", f"{vals[1]:.3f}", f"{vals[2]:.3f}", f"{vals[3]:.3f}", f"{vals[4]:.3f}",
            status
        ]
        md_content += "| " + " | ".join(row) + " |\n"
        
    with open(plots_dir / 'consolidated_radar_data.md', 'w') as f:
        f.write(md_content)
    print(f"✅ Generated consolidated_radar_data.md")

def run_consolidated_analysis():
    if not os.path.exists('master_model_comparison.csv'):
        return False
        
    df = pd.read_csv('master_model_comparison.csv')
    df['Model'] = df['Model'].replace(label_mapping)
    
    radar_data = get_global_radar_data(df)
    if radar_data:
        best_cent, best_fl = draw_global_radar(radar_data)
        export_markdown_table(radar_data, best_cent, best_fl)
        return True
    return False

if __name__ == "__main__":
    print("🚀 Running Consolidated Global Trade-off Analysis...")
    if run_consolidated_analysis():
        print("🎉 Global Analysis Complete!")
