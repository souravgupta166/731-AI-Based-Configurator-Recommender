from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/ml/v7_10_algorithms/731_v7_per_rfq_ranking_results.csv"
OUTPUT_DIR = ROOT / "data/processed/ml/v7_10_algorithms"

REGIME = "REALISTIC_BASELINE"
REFERENCE = "HIST_GRADIENT_BOOSTING"

METRICS = [
    "mrr",
    "ndcg3",
    "ndcg5",
    "ndcg10",
    "true_rank",
]

BOOTSTRAP_REPS = 5000
SEED = 731


def paired_bootstrap_difference(a, b, rng):
    """
    Difference = reference - comparison.
    Positive is better for MRR/NDCG.
    For true rank, negative is better.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    observed = np.mean(a - b)

    n = len(a)
    values = np.empty(BOOTSTRAP_REPS)

    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        values[i] = np.mean(
            a[idx] - b[idx]
        )

    return (
        observed,
        np.quantile(values, 0.025),
        np.quantile(values, 0.975),
    )


def paired_top1_test(reference, comparison):
    """
    Exact paired test for Top-1 success/failure.

    Only discordant RFQs matter:
        reference correct, comparison wrong
        reference wrong, comparison correct
    """
    reference = np.asarray(reference, dtype=int)
    comparison = np.asarray(comparison, dtype=int)

    ref_only = int(
        np.sum(
            (reference == 1)
            & (comparison == 0)
        )
    )

    comp_only = int(
        np.sum(
            (reference == 0)
            & (comparison == 1)
        )
    )

    discordant = ref_only + comp_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            ref_only,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return (
        ref_only,
        comp_only,
        discordant,
        p_value,
    )


def holm_adjust(p_values):
    """
    Holm step-down multiplicity correction.
    """
    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def main():

    df = pd.read_csv(INPUT)

    # Final statistical inference is performed only on the
    # held-out TEST RFQs. Validation RFQs are excluded.
    df = df[
        (df["regime"] == REGIME)
        & (df["split"] == "TEST")
    ].copy()

    models = sorted(
        df["model"].unique()
    )

    if REFERENCE not in models:
        raise ValueError(
            f"{REFERENCE} not found."
        )

    reference = df[
        df["model"] == REFERENCE
    ].copy()

    reference = reference.sort_values(
        "rfq_id"
    )

    results = []

    rng = np.random.default_rng(SEED)

    for model in models:

        if model == REFERENCE:
            continue

        comparison = df[
            df["model"] == model
        ].copy()

        comparison = comparison.sort_values(
            "rfq_id"
        )

        # Some model/regime combinations may contain duplicate
        # per-RFQ records in the exported V7 results. The ranking
        # result is defined at RFQ level, so retain one record per
        # RFQ after verifying that duplicates are identical.
        duplicate_counts = (
            comparison.groupby("rfq_id")
            .size()
        )

        duplicated_rfqs = duplicate_counts[
            duplicate_counts > 1
        ]

        if len(duplicated_rfqs) > 0:

            check_columns = [
                "top1",
                "top3",
                "top5",
                "top10",
                "mrr",
                "ndcg3",
                "ndcg5",
                "ndcg10",
                "true_rank",
            ]

            for rfq_id in duplicated_rfqs.index:

                rows = comparison[
                    comparison["rfq_id"] == rfq_id
                ]

                for column in check_columns:
                    if rows[column].nunique(
                        dropna=False
                    ) > 1:
                        raise ValueError(
                            f"{model}: duplicate RFQ {rfq_id} "
                            f"has conflicting values in {column}."
                        )

            comparison = (
                comparison
                .drop_duplicates(
                    subset=["rfq_id"],
                    keep="first",
                )
            )

        comparison = comparison.sort_values(
            "rfq_id"
        )

        merged = reference.merge(
            comparison,
            on="rfq_id",
            suffixes=("_ref", "_cmp"),
            validate="one_to_one",
        )

        if len(merged) != 732:
            raise ValueError(
                f"{model}: expected 732 paired RFQs, "
                f"found {len(merged)}."
            )

        # ---------------------------------------------------------
        # TOP-1: EXACT PAIRED BINOMIAL TEST
        # ---------------------------------------------------------

        ref_top1 = merged["top1_ref"].astype(int)
        cmp_top1 = merged["top1_cmp"].astype(int)

        (
            ref_only,
            cmp_only,
            discordant,
            top1_p,
        ) = paired_top1_test(
            ref_top1,
            cmp_top1,
        )

        # ---------------------------------------------------------
        # OTHER RANKING METRICS
        # ---------------------------------------------------------

        for metric in METRICS:

            ref_values = merged[
                f"{metric}_ref"
            ].to_numpy(dtype=float)

            cmp_values = merged[
                f"{metric}_cmp"
            ].to_numpy(dtype=float)

            difference = (
                ref_values - cmp_values
            )

            mean_difference = np.mean(
                difference
            )

            median_difference = np.median(
                difference
            )

            nonzero = difference[
                ~np.isclose(
                    difference,
                    0.0,
                )
            ]

            if len(nonzero) >= 2:

                try:
                    wilcoxon_result = wilcoxon(
                        ref_values,
                        cmp_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )

                    wilcoxon_p = (
                        wilcoxon_result.pvalue
                    )

                except ValueError:
                    wilcoxon_p = 1.0

            else:
                wilcoxon_p = 1.0

            (
                bootstrap_difference,
                ci_lower,
                ci_upper,
            ) = paired_bootstrap_difference(
                ref_values,
                cmp_values,
                rng,
            )

            results.append({
                "regime": REGIME,
                "reference_model": REFERENCE,
                "comparison_model": model,
                "metric": metric,
                "reference_mean": np.mean(
                    ref_values
                ),
                "comparison_mean": np.mean(
                    cmp_values
                ),
                "mean_difference_reference_minus_comparison":
                    mean_difference,
                "median_difference":
                    median_difference,
                "paired_bootstrap_difference":
                    bootstrap_difference,
                "ci_lower_95":
                    ci_lower,
                "ci_upper_95":
                    ci_upper,
                "wilcoxon_p":
                    wilcoxon_p,
                "top1_reference_only":
                    ref_only,
                "top1_comparison_only":
                    cmp_only,
                "top1_discordant":
                    discordant,
                "top1_exact_p":
                    top1_p,
            })

        print(
            f"{model:28s} "
            f"Top1 discordant="
            f"{discordant:3d} "
            f"Top1 p="
            f"{top1_p:.6f}"
        )

    result_df = pd.DataFrame(results)

    # -------------------------------------------------------------
    # MULTIPLE-COMPARISON CORRECTION
    # -------------------------------------------------------------

    # Holm correction separately for each metric.
    result_df["wilcoxon_p_holm"] = np.nan

    for metric in result_df["metric"].unique():

        mask = (
            result_df["metric"] == metric
        )

        result_df.loc[
            mask,
            "wilcoxon_p_holm",
        ] = holm_adjust(
            result_df.loc[
                mask,
                "wilcoxon_p",
            ].to_numpy()
        )

    # Top-1 has one test per comparison.
    top1_mask = (
        result_df["metric"] == "mrr"
    )

    top1_p_values = (
        result_df.loc[
            top1_mask,
            "top1_exact_p",
        ]
        .drop_duplicates()
        .to_numpy()
    )

    top1_adjusted = holm_adjust(
        top1_p_values
    )

    # Map adjusted Top-1 p-values by comparison.
    top1_pairs = result_df.loc[
        top1_mask,
        [
            "comparison_model",
            "top1_exact_p",
        ],
    ].drop_duplicates()

    top1_map = dict(
        zip(
            top1_pairs["comparison_model"],
            top1_adjusted,
        )
    )

    result_df["top1_exact_p_holm"] = (
        result_df["comparison_model"]
        .map(top1_map)
    )

    output = (
        OUTPUT_DIR
        / "731_v7_paired_statistical_comparisons.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    # -------------------------------------------------------------
    # COMPACT THESIS TABLE
    # -------------------------------------------------------------

    mrr = result_df[
        result_df["metric"] == "mrr"
    ].copy()

    compact = mrr[
        [
            "comparison_model",
            "reference_mean",
            "comparison_mean",
            "mean_difference_reference_minus_comparison",
            "ci_lower_95",
            "ci_upper_95",
            "wilcoxon_p",
            "wilcoxon_p_holm",
            "top1_exact_p_holm",
        ]
    ].copy()

    compact = compact.sort_values(
        "wilcoxon_p_holm"
    )

    compact.to_csv(
        OUTPUT_DIR
        / "731_v7_hgb_statistical_summary.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("HGB PAIRED STATISTICAL COMPARISON")
    print("=" * 80)

    print(
        compact.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(output)
    print(
        OUTPUT_DIR
        / "731_v7_hgb_statistical_summary.csv"
    )


if __name__ == "__main__":
    main()
