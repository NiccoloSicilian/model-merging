"""Compare weight update deltas between Dual and AdamW finetuning.

For each step_i.pt:
  1. Load dual delta from dual_dir
  2. Load adamw delta from adamw_dir
  3. Apply the duality map to the adamw delta
  4. Compute ||dual_delta - dualized_adamw_delta||_F for each layer and total

Usage:
    python scripts/compare_updates.py --dual_dir /path/to/dual --adamw_dir /path/to/adamw --output results.txt
    python scripts/compare_updates.py --dual_dir /path/to/dual --adamw_dir /path/to/adamw --output results.txt --mass_schedule linear --model B-32
"""

import argparse
import os
import re

import torch

from model_merging.merging.dual_arithmetic import ViT_B_16, ViT_B_32, ViT_L_14

# Reuse helpers from the classifier
from model_merging.model.dual_image_classifier import (
    build_duality_map_with_module,
    get_vit_topological_order,
)


def get_step_files(directory):
    """Return sorted list of (step_number, filepath) from a checkpoint dir."""
    files = []
    for f in os.listdir(directory):
        match = re.match(r"step_(\d+)\.pt", f)
        if match:
            files.append((int(match.group(1)), os.path.join(directory, f)))
    return sorted(files, key=lambda x: x[0])


def build_module(model_name, mass_schedule):
    if "B-16" in model_name:
        return ViT_B_16(mass_schedule=mass_schedule)
    elif "B-32" in model_name:
        return ViT_B_32(mass_schedule=mass_schedule)
    elif "L-14" in model_name:
        return ViT_L_14(mass_schedule=mass_schedule)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Compare Dual vs Dualized-AdamW updates")
    parser.add_argument("--dual_dir", required=True, help="Dir with dual finetuning step_*.pt")
    parser.add_argument("--adamw_dir", required=True, help="Dir with adamw finetuning step_*.pt")
    parser.add_argument("--output", required=True, help="Output results file")
    parser.add_argument("--mass_schedule", default="uniform", choices=["uniform", "linear"])
    parser.add_argument("--model", default="B-16", choices=["B-16", "B-32", "L-14"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dual_files = get_step_files(args.dual_dir)
    adamw_files = get_step_files(args.adamw_dir)

    print(f"Found {len(dual_files)} dual files, {len(adamw_files)} adamw files", flush=True)

    assert len(dual_files) == len(adamw_files), (
        f"Mismatch: {len(dual_files)} dual files vs {len(adamw_files)} adamw files"
    )

    print(f"Building duality module for {args.model}...", flush=True)
    module = build_module(args.model, args.mass_schedule)
    print(f"Module built. Atoms: {module.atoms}", flush=True)
    device = args.device
    print(f"Device: {device}", flush=True)

    results = []
    for (step_d, dual_path), (step_a, adamw_path) in zip(dual_files, adamw_files):
        assert step_d == step_a, f"Step mismatch: dual={step_d}, adamw={step_a}"
        print(f"Processing step {step_d}...", flush=True)

        print(f"  Loading dual: {dual_path}", flush=True)
        dual_delta = torch.load(dual_path, map_location=device)
        print(f"  Loading adamw: {adamw_path}", flush=True)
        adamw_delta = torch.load(adamw_path, map_location=device)
        print(f"  Loaded checkpoints ({len(dual_delta)} keys, {len(adamw_delta)} keys)", flush=True)

        # Cast to float32 (checkpoints are float16)
        dual_delta = {k: v.float() for k, v in dual_delta.items()}
        adamw_delta = {k: v.float() for k, v in adamw_delta.items()}
        print(f"  Cast to float32", flush=True)

        # Sort and apply duality map to adamw deltas
        layer_names = get_vit_topological_order(list(adamw_delta.keys()))
        print(f"  Sorted {len(layer_names)} layers, applying duality map...", flush=True)

        dualized_adamw = build_duality_map_with_module(
            layer_names=layer_names,
            grads=adamw_delta,
            device=device,
            mass_schedule=args.mass_schedule,
            model_name=args.model,
            m=module,
        )
        print(f"  Duality map applied, computing norms...", flush=True)

        # Compute per-layer Frobenius norm and cosine similarity
        layer_norms = {}
        layer_cosines = {}
        for name in dualized_adamw:
            if name in dual_delta:
                a = dual_delta[name]
                b = dualized_adamw[name]
                diff = a - b
                layer_norms[name] = torch.norm(diff, p="fro").item()
                cos = torch.nn.functional.cosine_similarity(
                    a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
                ).item()
                layer_cosines[name] = cos

        results.append((step_d, layer_norms, layer_cosines))
        avg_norm = sum(layer_norms.values()) / len(layer_norms)
        avg_cos = sum(layer_cosines.values()) / len(layer_cosines)
        print(f"  step {step_d} done: {len(layer_norms)} layers, avg_norm={avg_norm:.6f}, avg_cosine={avg_cos:.6f}", flush=True)

    with open(args.output, "w") as f:
        if results:
            layer_names_sorted = sorted(results[0][1].keys())
            # Header: norm columns then cosine columns
            norm_cols = [f"norm_{n}" for n in layer_names_sorted]
            cos_cols = [f"cos_{n}" for n in layer_names_sorted]
            f.write("step\tavg_norm\tavg_cosine\t" + "\t".join(norm_cols + cos_cols) + "\n")
            for step, layer_norms, layer_cosines in results:
                avg_n = sum(layer_norms.values()) / len(layer_norms)
                avg_c = sum(layer_cosines.values()) / len(layer_cosines)
                norm_vals = "\t".join(f"{layer_norms[n]:.6f}" for n in layer_names_sorted)
                cos_vals = "\t".join(f"{layer_cosines[n]:.6f}" for n in layer_names_sorted)
                f.write(f"{step}\t{avg_n:.6f}\t{avg_c:.6f}\t{norm_vals}\t{cos_vals}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
