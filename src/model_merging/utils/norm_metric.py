import torch
import os
import json
from math import sqrt
def save_latex_plot(x_vals, y_vals, best_alpha, alpha_fix, output_tex, title_prefix=""):
    """Generates a standalone LaTeX file with a pgfplots graph."""
    coordinates = "\n".join([f"            ({x:.4f}, {y:.6f})" for x, y in zip(x_vals, y_vals)])

    # Handle summary plot where best_alpha is not applicable
    if best_alpha is not None:
        title_str = f"{title_prefix} Distance ($\\\\alpha_f = {alpha_fix}$) \\\\\\\\ Best $\\\\alpha_c$: {best_alpha:.2f}"
    else:
        title_str = f"{title_prefix} ($\\\\alpha_f = {alpha_fix}$)"

    latex_code = f"""\\documentclass{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}

\\begin{{document}}
\\begin{{tikzpicture}}
    \\begin{{axis}}[
        title={{{title_str}}},
        title style={{align=center}},
        xlabel={{$\\alpha_c$}},
        ylabel={{Avg Frobenius Distance}},
        grid=major,
        width=10cm,
        height=7cm,
        legend pos=north east,
    ]
        \\addplot[
            color=blue,
            mark=*,
            mark size=1.5pt
        ] coordinates {{
{coordinates}
        }};
    \\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    os.makedirs(os.path.dirname(output_tex) if os.path.dirname(output_tex) else ".", exist_ok=True)

    with open(output_tex, "w") as f:
        f.write(latex_code)
    print(f"\n✅ LaTeX plot saved to {os.path.abspath(output_tex)}")

def evaluate_alphasWeighted(file_compare, file_fixed, alpha_fix=1.0, n_task=8, output_tex="alpha_plot.tex"):
    print(f"Loading files:\n1. {file_compare}\n2. {file_fixed}... for tasks {n_task}")
    
    dict_compare = torch.load(file_compare, map_location='cpu')
    dict_fixed = torch.load(file_fixed, map_location='cpu')
    
    layers_to_exclude = {
        'model.positional_embedding',
        'model.text_projection',
        'model.token_embedding.weight'
    }

    common_keys = [
        k for k in dict_compare.keys() 
        if k in dict_fixed.keys()
        and 'conv' not in k.lower() 
        and dict_compare[k].dim() == 2
        and k not in layers_to_exclude
    ]
    
    if not common_keys:
        print("❌ Error: No matching layers found.")
        return
    
    print(f"Found {len(common_keys)} matching layers.")

    mc_dict = {}
    mf_scaled_dict = {}
    for k in common_keys:
        mc_dict[k] = dict_compare[k].float()
        mf_scaled_dict[k] =  dict_fixed[k].float()

    valid_keys = list(mc_dict.keys())
    n = len(valid_keys)
    print(f"Valid layers: {n}/{len(common_keys)}")
    print("\nIterating alpha_c from 0.1 to 2.0")
    print("-" * 60)

    alpha_vals = []
    distance_vals = []
    c_fixed = {}
    s_compare = {}
    
    for k in valid_keys:
        m_fixed = mf_scaled_dict[k]
        m_compare = mc_dict[k]
        s_fixed = torch.linalg.svdvals(m_fixed)
        c_fixed[k] = s_fixed
        s_compare[k] = torch.linalg.svdvals(m_compare)
        
    for step in range(1, 35):
        alpha_c = step / 10.0
        total_distance = 0.0
        
        for k in valid_keys:
            s_comp = s_compare[k]
            c = c_fixed[k]
            
            # Keep only first 1/n_task of singular values
            k_top = max(1, len(s_comp) // n_task)
            s_comp_top = s_comp[:k_top]
            c_top = c[:k_top]
            
            weights = s_comp_top**2
            diffs = torch.abs(alpha_c * s_comp_top - alpha_fix*c_top)**2
            weighted_avg = (diffs*weights).sum()/weights.sum()
            total_distance += weighted_avg.item()

        avg_dist = total_distance / n
        print(f"{alpha_c} | {avg_dist}")
        alpha_vals.append(alpha_c)
        distance_vals.append(avg_dist)

    best_alpha = alpha_vals[distance_vals.index(min(distance_vals))]
    print(f"\n🎯 Best alpha_c: {best_alpha:.1f}")

    if alpha_vals:
        save_latex_plot(alpha_vals, distance_vals, best_alpha, alpha_fix, output_tex, title_prefix="TA Weighted L1")

def evaluate_alphasNonWeighted(file_compare, file_fixed, alpha_fix=1.0, n_task=8, output_tex="alpha_plot.tex"):
    print(f"Loading files:\n1. {file_compare}\n2. {file_fixed}... for tasks {n_task}")
    
    dict_compare = torch.load(file_compare, map_location='cpu')
    dict_fixed = torch.load(file_fixed, map_location='cpu')
    
    layers_to_exclude = {
        'model.positional_embedding',
        'model.text_projection',
        'model.token_embedding.weight'
    }

    common_keys = [
        k for k in dict_compare.keys() 
        if k in dict_fixed.keys()
        and 'conv' not in k.lower() 
        and dict_compare[k].dim() == 2
        and k not in layers_to_exclude
    ]
    
    if not common_keys:
        print("❌ Error: No matching layers found.")
        return
    
    print(f"Found {len(common_keys)} matching layers.")

    mc_dict = {}
    mf_scaled_dict = {}
    for k in common_keys:
        mc_dict[k] = dict_compare[k].float()
        mf_scaled_dict[k] =  dict_fixed[k].float()

    valid_keys = list(mc_dict.keys())
    n = len(valid_keys)
    print(f"Valid layers: {n}/{len(common_keys)}")
    print("\nIterating alpha_c from 0.1 to 2.0")
    print("-" * 60)

    alpha_vals = []
    distance_vals = []
    c_fixed = {}
    s_compare = {}
    
    for k in valid_keys:
        m_fixed = mf_scaled_dict[k]
        m_compare = mc_dict[k]
        s_fixed = torch.linalg.svdvals(m_fixed)
        c_fixed[k] = s_fixed
        s_compare[k] = torch.linalg.svdvals(m_compare)
        
    for step in range(1, 35):
        alpha_c = step / 10.0
        total_distance = 0.0
        
        for k in valid_keys:
            s_comp = s_compare[k]
            c = c_fixed[k]
            
            # Keep only first 1/n_task of singular values
            k_top = max(1, len(s_comp) // n_task)
            s_comp_top = s_comp[:k_top]
            c_top = c[:k_top]
            
            
            diffs = torch.abs(alpha_c * s_comp_top - alpha_fix*c_top)**2
            avg = (diffs).sum()/len(s_comp_top)
            total_distance += avg.item()

        avg_dist = total_distance / n
        print(f"{alpha_c} | {avg_dist}")
        alpha_vals.append(alpha_c)
        distance_vals.append(avg_dist)

    best_alpha = alpha_vals[distance_vals.index(min(distance_vals))]
    print(f"\n🎯 Best alpha_c: {best_alpha:.1f}")

    if alpha_vals:
        save_latex_plot(alpha_vals, distance_vals, best_alpha, alpha_fix, output_tex, title_prefix="TA Weighted L1")

def evaluate_alphasNonWeightedVT(file_compare, file_fixed, alpha_fix=1.0, n_tasks_list=None, output_tex="alpha_plot.tex"):
    if n_tasks_list is None:
        n_tasks_list = list(range(1, 21))

    print(f"Loading files:\n1. {file_compare}\n2. {file_fixed}")

    dict_compare = torch.load(file_compare, map_location='cpu')
    dict_fixed = torch.load(file_fixed, map_location='cpu')

    layers_to_exclude = {
        'model.positional_embedding',
        'model.text_projection',
        'model.token_embedding.weight'
    }
    common_keys = [
        k for k in dict_compare.keys()
        if k in dict_fixed.keys()
        and 'conv' not in k.lower()
        and dict_compare[k].dim() == 2
        and k not in layers_to_exclude
    ]

    if not common_keys:
        print("❌ Error: No matching layers found.")
        return

    print(f"Found {len(common_keys)} matching layers.")

    mc_dict = {}
    mf_scaled_dict = {}
    for k in common_keys:
        mc_dict[k] = dict_compare[k].float()
        mf_scaled_dict[k] = dict_fixed[k].float()

    valid_keys = list(mc_dict.keys())
    n = len(valid_keys)
    print(f"Valid layers: {n}/{len(common_keys)}")

    # Precompute singular values once for all layers
    c_fixed = {}
    s_compare = {}
    for k in valid_keys:
        c_fixed[k] = torch.linalg.svdvals(mf_scaled_dict[k])
        s_compare[k] = torch.linalg.svdvals(mc_dict[k])

    # Results across all n_task values
    best_alphas = {}

    for n_task in n_tasks_list:
        print(f"\n{'=' * 60}")
        print(f"📌 n_task = {n_task}")
        print(f"{'=' * 60}")
        print(f"{'alpha_c':>10} | {'avg_distance':>15}")
        print("-" * 60)

        alpha_vals = []
        distance_vals = []

        for step in range(1, 35):
            alpha_c = step / 10.0
            total_distance = 0.0

            for k in valid_keys:
                s_comp = s_compare[k]
                c = c_fixed[k]

                k_top = max(1, len(s_comp) // n_task)
                s_comp_top = s_comp[:k_top]
                c_top = c[:k_top]

                # Calculate the difference tensor
                diff_tensor = alpha_c * s_comp_top - alpha_fix * c_top

                # Use PyTorch's built-in norm function (calculates true Frobenius norm)
                frob_norm = torch.linalg.norm(diff_tensor)

                # Add to total distance
                total_distance += frob_norm.item()**2

            avg_dist = total_distance / n
            print(f"{alpha_c:>10.2f} | {avg_dist:>15.6f}")
            alpha_vals.append(alpha_c)
            distance_vals.append(avg_dist)

        best_alpha = alpha_vals[distance_vals.index(min(distance_vals))]
        best_alphas[n_task] = best_alpha
        print(f"\n🎯 Best alpha_c for n_task={n_task}: {best_alpha:.2f}")

        if alpha_vals:
            tex_name = output_tex.replace(".tex", f"_ntask{n_task}.tex")
            #save_latex_plot(alpha_vals, distance_vals, best_alpha, alpha_fix, tex_name, title_prefix=f"TA Weighted L1 (n_task={n_task})")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"📊 Summary: Best alpha per number of tasks")
    print(f"{'=' * 60}")
    print(f"{'n_task':>10} | {'best_alpha':>12}")
    print("-" * 60)
    for n_task, alpha in sorted(best_alphas.items()):
        print(f"{n_task:>10} | {alpha:>12.2f}")

    # Save results to JSON
    results_path = output_tex.replace(".tex", "_best_alphas.json")
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in best_alphas.items()}, f, indent=2)
    print(f"\n💾 Best alphas saved to {results_path}")

    # Save summary plot: x = n_tasks, y = best_alpha
    sorted_tasks = sorted(best_alphas.keys())
    sorted_alphas = [best_alphas[t] for t in sorted_tasks]
    summary_tex_path = output_tex.replace(".tex", "_summary.tex")
    save_latex_plot(
        sorted_tasks,
        sorted_alphas,
        best_alpha=None,
        alpha_fix=alpha_fix,
        output_tex=summary_tex_path,
        title_prefix="Optimal Alpha vs Number of Tasks",
    )
    print(f"📊 Summary plot saved to {summary_tex_path}")

    return best_alphas

if __name__ == "__main__":
    file1 = "./results/ISO/matrixesISOC_ViT-B-16task14.pt"
    file2 = "./results/DUAL/matrixesDual_ViT-B-16task14.pt"
    n_task = 1
    alpha_fix = 1.2 # Note the change in extension here to .tex
    #evaluate_alphasTSV(file1, file2, alpha_fix=alpha_fix, output_tex="outputs/alpha_plot_TSV.tex")
    evaluate_alphasNonWeightedVT(file1, file2, alpha_fix,n_tasks_list=list(range(1, 99)), output_tex="outputs/alpha_comparison_plot.tex")
    #evaluate_alphasNonWeighted(file1, file2, alpha_fix,n_task=n_task, output_tex="outputs/alpha_comparison_plot.tex")
    #evaluate_min_vs_mean_distance(file1, file2, output_tex="alpha_plot_min_mean.tex")
    #evaluate_alphasMY(file1, file2, alpha_fix, n_task=n_task, output_tex="outputs/alpha_comparison_plot.tex")