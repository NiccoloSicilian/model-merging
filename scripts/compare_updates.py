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

    module = build_module(args.model, args.mass_schedule)
    device = args.device

    results = []
    for (step_d, dual_path), (step_a, adamw_path) in zip(dual_files, adamw_files):
        assert step_d == step_a, f"Step mismatch: dual={step_d}, adamw={step_a}"

        dual_delta = torch.load(dual_path, map_location=device)
        adamw_delta = torch.load(adamw_path, map_location=device)

        # Cast to float32 (checkpoints are float16)
        dual_delta = {k: v.float() for k, v in dual_delta.items()}
        adamw_delta = {k: v.float() for k, v in adamw_delta.items()}

        # Sort and apply duality map to adamw deltas
        layer_names = get_vit_topological_order(list(adamw_delta.keys()))

        dualized_adamw = build_duality_map_with_module(
            layer_names=layer_names,
            grads=adamw_delta,
            device=device,
            mass_schedule=args.mass_schedule,
            model_name=args.model,
            m=module,
        )

        # Compute per-layer Frobenius norm of (dual - dualized_adamw)
        layer_norms = {}
        for name in dualized_adamw:
            if name in dual_delta:
                diff = dual_delta[name] - dualized_adamw[name]
                layer_norms[name] = torch.norm(diff, p="fro").item()

        results.append((step_d, layer_norms))
        print(f"step {step_d}: {len(layer_norms)} layers compared", flush=True)

    with open(args.output, "w") as f:
        # Header
        if results:
            layer_names_sorted = sorted(results[0][1].keys())
            f.write("step\tavg_norm\t" + "\t".join(layer_names_sorted) + "\n")
            for step, layer_norms in results:
                avg = sum(layer_norms.values()) / len(layer_norms)
                values = "\t".join(f"{layer_norms[n]:.6f}" for n in layer_names_sorted)
                f.write(f"{step}\t{avg:.6f}\t{values}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
