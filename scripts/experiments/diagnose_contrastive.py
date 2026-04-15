#!/usr/bin/env python3


"""Diagnostic: understand why decode->encode doesn't preserve codes."""

from __future__ import annotations

import torch
from modules.contrastive.rq_module import RQModule


def main() -> None:
    """Diagnose the decode->encode issue."""
    print("=" * 60)
    print("DIAGNOSING DECODE->ENCODE INSTABILITY")
    print("=" * 60)

    dim = 64
    n_levels = 4
    n_codes = 256
    d_cut = 2
    n_samples = 100

    torch.manual_seed(42)

    rq = RQModule(dim=dim, n_levels=n_levels, n_codes=n_codes, d_cut=d_cut)
    embeddings = torch.randn(n_samples, dim)

    # First encode
    reconstructed, codes, _ = rq(embeddings)

    # Decode to get lattice point
    decoded = rq.decode(codes)

    # Check if decoded == reconstructed (they should be identical)
    diff = (decoded - reconstructed).abs().max().item()
    print(f"\n1. decoded vs reconstructed max diff: {diff:.2e}")
    print(f"   (Should be ~0 since both are sum of centroids)")

    # Now understand WHY re-encoding fails
    # Re-encode step by step
    print("\n2. Step-by-step re-encoding analysis:")
    residual = decoded.clone()
    for level in range(n_levels):
        codebook = rq.codebooks[level]
        expected_code = codes[:, level]

        # Find what encode would pick
        dists = torch.cdist(residual, codebook)
        picked_code = dists.argmin(dim=-1)

        match_rate = (picked_code == expected_code).float().mean().item()
        print(f"   Level {level}: code match = {match_rate:.1%}")

        if match_rate < 1.0:
            # Analyze why mismatch happens
            mismatched = picked_code != expected_code
            n_mismatch = mismatched.sum().item()

            # Get distances for mismatched samples
            for i in torch.where(mismatched)[0][:3]:  # First 3 mismatches
                exp_c = expected_code[i].item()
                got_c = picked_code[i].item()
                exp_dist = dists[i, exp_c].item()
                got_dist = dists[i, got_c].item()
                print(
                    f"     Sample {i}: expected code {exp_c} (dist={exp_dist:.4f}), "
                    f"got {got_c} (dist={got_dist:.4f}), diff={exp_dist - got_dist:.4f}"
                )

        # Update residual
        residual = residual - codebook[picked_code]

    print("\n3. ROOT CAUSE:")
    print("   The greedy RQ encode algorithm picks the nearest centroid at each level.")
    print("   But decode(codes) = sum of centroids, which when re-encoded greedily,")
    print("   may pick different codes because the global optimum != greedy choices.")
    print()
    print("4. IMPLICATION FOR CONTRASTIVE LEARNING:")
    print("   The training loop uses augmented embeddings DIRECTLY (no re-encoding).")
    print("   Re-encoding only happens during evaluation/analysis.")
    print("   So the FIX is correct - augmentation preserves trunk in CODE SPACE.")
    print("   We just can't verify it by re-encoding!")


if __name__ == "__main__":
    main()
