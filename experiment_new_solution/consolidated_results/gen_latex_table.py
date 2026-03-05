import pandas as pd

df = pd.read_csv('/home/pqvinh/Documents/LABs/FedAvgLS/experiment_new_solution/consolidated_results/latex/plots/all_result.csv')

# Preprocessing: Simplify Paradigm names for space
df['Paradigm'] = df['Paradigm'].replace({'Centralized': 'Cent.', 'FL': 'FL'})
df['Task_Type'] = df['Task_Type'].replace({'Multi-Task (MTL)': 'MTL', 'Single-Task': 'ST'})

# Formatting function for LaTeX
def format_val(v, is_max=False, is_suggested=False):
    if pd.isna(v) or v == '':
        return 'N/A'
    try:
        val = f"{float(v):.4f}"
        if is_max:
            val = f"\\textbf{{{val}}}"
        if is_suggested:
            val = f"\\textcolor{{red}}{{{val}}}"
        return val
    except:
        return str(v)

print("\\begin{table*}[t]")
print("  \\centering")
print("  \\caption{Advanced Comparative Analysis of SLM Performance across Paradigms}")
print("  \\label{tab:results_consolidated}")
print("  \\scriptsize")
print("  \\setlength{\\tabcolsep}{3pt}")
print("  \\begin{tabular}{lllllcccccccc}")
print("    \\toprule")
print("    & & & & & \\multicolumn{2}{c}{\\textbf{SST-2}} & \\multicolumn{2}{c}{\\textbf{QQP}} & \\multicolumn{2}{c}{\\textbf{STS-B}} & \\textbf{System} & \\textbf{System} \\\\")
print("    \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}")
print("    \\textbf{Model} & \\textbf{Paradigm} & \\textbf{Task} & \\textbf{Dist.} & \\textbf{Tuning} & \\textbf{Acc.} & \\textbf{F1} & \\textbf{Acc.} & \\textbf{F1} & \\textbf{Pear.} & \\textbf{Spea.} & \\textbf{Time (s)} & \\textbf{Res (GB)} \\\\")
print("    \\midrule")

# Sort: Model, Paradigm, Task
model_order = ['DistilBERT', 'BERT-Medium', 'BERT-Mini', 'MiniLM', 'TinyBERT']
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
df = df.sort_values(['Model', 'Paradigm', 'Task_Type'], ascending=[True, False, False])

# Identify max values for highlighting (primary metrics only)
metrics = ['Val_SST2_Acc', 'Val_SST2_F1', 'Val_QQP_Acc', 'Val_QQP_F1', 'Val_STSB_Pearson', 'Val_STSB_Spearman']
max_vals = {col: df[col].max() for col in metrics}

models = df['Model'].unique()
for model in models:
    model_df = df[df['Model'] == model]
    first_row = True
    for _, row in model_df.iterrows():
        # Display model name only on the first row of its group
        model_display = f"\\textbf{{{model}}}" if first_row else ""
        
        # Highlight our method (FL-MTL)
        paradigm_val = row['Paradigm']
        task_val = row['Task_Type']
        is_ours = paradigm_val == 'FL' and task_val == 'MTL'
        if is_ours:
            task_val = "\\textbf{MTL (ours)}"
        
        # Suggested Model Highlight (DistilBERT + FL + MTL + Non-IID)
        is_suggested = is_ours and model == 'DistilBERT' and row['Distribution'] == 'Non-IID'
        
        if is_suggested:
            model_display = f"\\textcolor{{red}}{{{model_display}}}" if first_row else ""
            paradigm_val = f"\\textcolor{{red}}{{{paradigm_val}}}"
            task_val = f"\\textcolor{{red}}{{{task_val}}}"
            dist_val = f"\\textcolor{{red}}{{{row['Distribution']}}}"
            tuning_val = f"\\textcolor{{red}}{{{row['Tuning_Method']}}}"
        else:
            dist_val = row['Distribution']
            tuning_val = row['Tuning_Method']

        line = [
            model_display,
            paradigm_val,
            task_val,
            dist_val,
            tuning_val,
            format_val(row['Val_SST2_Acc'], row['Val_SST2_Acc'] == max_vals['Val_SST2_Acc'], is_suggested),
            format_val(row['Val_SST2_F1'], row['Val_SST2_F1'] == max_vals['Val_SST2_F1'], is_suggested),
            format_val(row['Val_QQP_Acc'], row['Val_QQP_Acc'] == max_vals['Val_QQP_Acc'], is_suggested),
            format_val(row['Val_QQP_F1'], row['Val_QQP_F1'] == max_vals['Val_QQP_F1'], is_suggested),
            format_val(row['Val_STSB_Pearson'], row['Val_STSB_Pearson'] == max_vals['Val_STSB_Pearson'], is_suggested),
            format_val(row['Val_STSB_Spearman'], row['Val_STSB_Spearman'] == max_vals['Val_STSB_Spearman'], is_suggested),
            f"\\textcolor{{red}}{{{row['Total_Train_Time']:.2f}}}" if is_suggested else f"{row['Total_Train_Time']:.2f}",
            f"\\textcolor{{red}}{{{row['Resource_Usage']:.4f}}}" if is_suggested else f"{row['Resource_Usage']:.4f}"
        ]
        print("    " + " & ".join(line) + " \\\\")
        first_row = False
    print("    \\midrule")

print("    \\bottomrule")
print("    \\addlinespace[2pt]")
print("    \\multicolumn{13}{l}{\\textit{Note: Bold values indicate highest performance. N/A = Not Applicable. Abbreviations: Dist. = Data Distribution,}} \\\\")
print("    \\multicolumn{13}{l}{\\textit{ST = Single-Task, MTL = Multi-Task Learning, Cent. = Centralized, LoRA = Low-Rank Adaptation,}} \\\\")
print("    \\multicolumn{13}{l}{\\textit{FFT = Full Fine-Tuning, Pear. = Pearson Correlation, Spea. = Spearman Correlation.}} \\\\")
print("  \\end{tabular}")
print("\\end{table*}")
