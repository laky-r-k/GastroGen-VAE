"""Experiment 12: paired significance analysis of saved 70/30 model results."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from experiments.base_experiment import BaseExperiment


class Experiment12(BaseExperiment):
    """Analyze matched per-seed AUC differences without retraining models."""

    experiment_name = "experiment12"

    def _read_metrics(self, path: str, experiment: str) -> pd.DataFrame:
        full_path = self._resolve_path(path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Missing {experiment} per-seed metrics: {full_path}")
        frame = pd.read_csv(full_path)
        required = {"seed", "roc_auc"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{experiment} metrics lack columns: {sorted(missing)}")
        return frame

    @staticmethod
    def _norm_ids(values) -> set:
        return {str(int(float(value))) for value in values}

    def _validate_manifests(self, manifest_paths: Dict[str, str], seeds: List[int]) -> None:
        manifests = {}
        for name, path in manifest_paths.items():
            full_path = self._resolve_path(path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Missing {name} split manifest: {full_path}")
            frame = pd.read_csv(full_path)
            needed = {"seed", "patient_id", "partition"}
            if not needed.issubset(frame.columns):
                raise ValueError(f"{name} split manifest lacks {sorted(needed - set(frame.columns))}")
            manifests[name] = frame

        reference_name = next(iter(manifests))
        reference = manifests[reference_name]
        for seed in seeds:
            ref_seed = reference[reference.seed == seed]
            if ref_seed.empty:
                raise ValueError(f"{reference_name} manifest has no rows for seed {seed}")
            for name, frame in manifests.items():
                current = frame[frame.seed == seed]
                for partition in ("train", "test"):
                    ref_ids = self._norm_ids(
                        ref_seed.loc[ref_seed.partition == partition, "patient_id"]
                    )
                    current_ids = self._norm_ids(
                        current.loc[current.partition == partition, "patient_id"]
                    )
                    if not ref_ids or ref_ids != current_ids:
                        raise ValueError(
                            f"Split mismatch for seed {seed}, {partition}: "
                            f"{reference_name} versus {name}; refusing unpaired comparison."
                        )

    @staticmethod
    def _select_one(
        frame: pd.DataFrame,
        *,
        seed_values: List[int],
        model: str,
        beta=None,
        condition=None,
        exp_name: str,
    ) -> pd.Series:
        subset = frame[frame.seed.isin(seed_values)].copy()
        if "model" in subset:
            subset = subset[subset.model == model]
        elif "condition" in subset:
            subset = subset[subset.condition == condition]
        else:
            raise ValueError(f"{exp_name} metrics have neither model nor condition column")
        if beta is not None:
            if "beta" not in subset:
                raise ValueError(f"{exp_name} metrics do not contain beta")
            numeric_beta = pd.to_numeric(subset.beta, errors="coerce")
            subset = subset[np.isclose(numeric_beta, float(beta), rtol=0, atol=1e-12)]
        if subset.seed.duplicated().any():
            raise ValueError(f"{exp_name} selection {model}, beta={beta} has duplicate seed rows")
        result = subset.set_index("seed").roc_auc.sort_index()
        if list(result.index.astype(int)) != seed_values:
            missing = sorted(set(seed_values) - set(result.index.astype(int)))
            raise ValueError(
                f"{exp_name} selection {model}, beta={beta} does not have all expected seeds; "
                f"missing={missing}"
            )
        return result.astype(float)

    @staticmethod
    def _holm_adjust(p_values: List[float]) -> List[float]:
        values = np.asarray(p_values, dtype=float)
        order = np.argsort(values)
        adjusted_sorted = np.maximum.accumulate(
            (len(values) - np.arange(len(values))) * values[order]
        )
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        return adjusted.tolist()

    def run(self):
        cfg = self.config
        seed_cfg = cfg["seeds"]
        seeds = list(range(int(seed_cfg["start"]), int(seed_cfg["end"]) + 1))
        alpha = float(cfg.get("alpha", 0.05))
        if not seeds:
            raise ValueError("No seeds configured")

        inputs = cfg["inputs"]
        exp5 = self._read_metrics(inputs["experiment5_metrics"], "Experiment 5")
        exp7 = self._read_metrics(inputs["experiment7_metrics"], "Experiment 7")
        exp10 = self._read_metrics(inputs["experiment10_metrics"], "Experiment 10")
        exp11 = self._read_metrics(inputs["experiment11_metrics"], "Experiment 11")
        self._validate_manifests(inputs["split_manifests"], seeds)

        # Make the intended 70/30 holdout ratio explicit for the corrected
        # resampled t variance estimate.
        tabpfn_real_rows = exp11[
            (exp11.seed == seeds[0]) & (exp11.condition == "real_only")
        ]
        if len(tabpfn_real_rows) != 1:
            raise ValueError("Cannot determine holdout sizes from Experiment 11 metrics")
        train_n = int(tabpfn_real_rows.iloc[0].real_train_rows)
        test_n = int(tabpfn_real_rows.iloc[0].test_rows)
        test_train_ratio = test_n / train_n

        raw_gb = self._select_one(
            exp5, seed_values=seeds, model="Raw 8-feature GB", beta=None,
            exp_name="Experiment 5 raw GB baseline",
        )
        # The Experiment 5 raw GB baseline marks beta as the string "baseline".
        raw_gb_rows = exp5[(exp5.seed.isin(seeds)) & (exp5.model == "Raw 8-feature GB")]
        raw_gb = raw_gb_rows.set_index("seed").roc_auc.astype(float).sort_index()
        if list(raw_gb.index.astype(int)) != seeds or raw_gb.index.duplicated().any():
            raise ValueError("Experiment 5 raw GB baseline must have exactly one row per seed")

        paired_series = {
            "TabPFN real-only minus paper-config GB real-only": (
                self._select_one(
                    exp11, seed_values=seeds, model="", condition="real_only",
                    exp_name="Experiment 11 TabPFN real-only",
                ), raw_gb,
            ),
            "GB + WGAN minus GB real-only": (
                self._select_one(
                    exp10, seed_values=seeds, model="Raw 8-feature GB + WGAN-GP",
                    exp_name="Experiment 10 GB + WGAN",
                ), raw_gb,
            ),
            "VAE + GB + WGAN minus VAE + GB (beta=0.0005)": (
                self._select_one(
                    exp10, seed_values=seeds, model="VAE (z=4) + GB + WGAN-GP", beta=0.0005,
                    exp_name="Experiment 10 VAE + WGAN",
                ),
                self._select_one(
                    exp7, seed_values=seeds, model="VAE (z=4) + GB", beta=0.0005,
                    exp_name="Experiment 7 VAE real-only",
                ),
            ),
            "SS-VAE + GB + WGAN minus SS-VAE + GB (beta=0.0005)": (
                self._select_one(
                    exp10, seed_values=seeds,
                    model="SS-VAE (z=4) + GB + WGAN-GP", beta=0.0005,
                    exp_name="Experiment 10 SS-VAE + WGAN",
                ),
                self._select_one(
                    exp5, seed_values=seeds, model="SS-VAE (z=4) + GB", beta=0.0005,
                    exp_name="Experiment 5 SS-VAE real-only",
                ),
            ),
            "TabPFN real + WGAN minus TabPFN real-only": (
                self._select_one(
                    exp11, seed_values=seeds, model="", condition="real_plus_wgan",
                    exp_name="Experiment 11 TabPFN real + WGAN",
                ),
                self._select_one(
                    exp11, seed_values=seeds, model="", condition="real_only",
                    exp_name="Experiment 11 TabPFN real-only",
                ),
            ),
        }

        deltas_for_export = pd.DataFrame({"seed": seeds})
        rows = []
        correction = (1.0 / len(seeds)) + test_train_ratio
        for comparison, (model_a, model_b) in paired_series.items():
            if not model_a.index.equals(model_b.index):
                raise ValueError(f"Seed alignment failed for comparison: {comparison}")
            delta = model_a.to_numpy() - model_b.to_numpy()
            deltas_for_export[comparison] = delta
            mean_delta = float(np.mean(delta))
            sd_delta = float(np.std(delta, ddof=1))
            corrected_se = float(np.sqrt(correction * sd_delta**2))
            t_stat = mean_delta / corrected_se if corrected_se > 0 else 0.0
            p_value = float(2 * student_t.sf(abs(t_stat), df=len(seeds) - 1))
            critical_t = float(student_t.ppf(1 - alpha / 2, df=len(seeds) - 1))
            rows.append({
                "comparison": comparison,
                "n_seeds": len(seeds),
                "mean_auc_difference_a_minus_b": mean_delta,
                "sd_auc_difference": sd_delta,
                "corrected_standard_error": corrected_se,
                "corrected_ci_level": 1 - alpha,
                "corrected_ci_lower": mean_delta - critical_t * corrected_se,
                "corrected_ci_upper": mean_delta + critical_t * corrected_se,
                "test_train_ratio": test_train_ratio,
                "variance_correction_factor": correction,
                "t_statistic": t_stat,
                "degrees_freedom": len(seeds) - 1,
                "p_value_two_sided": p_value,
                "seeds_a_higher": int(np.sum(delta > 0)),
                "seeds_tied": int(np.sum(delta == 0)),
                "seeds_b_higher": int(np.sum(delta < 0)),
            })

        results = pd.DataFrame(rows)
        results["p_value_holm"] = self._holm_adjust(results.p_value_two_sided.tolist())
        results["significant_holm_alpha_0_05"] = results.p_value_holm < alpha

        metrics_dir = self.get_results_subdir("metrics")
        deltas_for_export.to_csv(os.path.join(metrics_dir, "paired_auc_differences.csv"), index=False)
        results.to_csv(os.path.join(metrics_dir, "corrected_significance_tests.csv"), index=False)

        wgan_rows = results[results.comparison.str.contains("WGAN")]
        if (wgan_rows.p_value_holm < alpha).any():
            wgan_conclusion = (
                "At least one WGAN comparison shows a statistically reliable difference after Holm adjustment. "
                "Use the signed AUC differences to state whether that comparison improved or reduced AUC."
            )
        else:
            wgan_conclusion = (
                "No statistically reliable evidence of an AUC improvement from WGAN augmentation was found "
                "in these four tested model comparisons. This does not prove that the true effect is exactly zero."
            )
        tabpfn_gb = results.iloc[0]
        if tabpfn_gb.p_value_holm < alpha and tabpfn_gb.mean_auc_difference_a_minus_b > 0:
            model_conclusion = "TabPFN real-only has significantly higher AUC than paper-config GB after Holm adjustment."
        elif tabpfn_gb.p_value_holm < alpha:
            model_conclusion = "Paper-config GB has significantly higher AUC than real-only TabPFN after Holm adjustment."
        else:
            model_conclusion = "The analysis does not show a statistically reliable AUC difference between real-only TabPFN and paper-config GB."

        report_path = os.path.join(self.run_dir, "analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Experiment 12: corrected paired AUC significance analysis\n\n")
            f.write(f"Analyzed {len(seeds)} matched stratified 70/30 splits (seeds {seeds[0]}–{seeds[-1]}).\n\n")
            f.write("No models were retrained. Split manifests were checked before the analysis; all comparisons use the same patient IDs in each train/test partition.\n\n")
            f.write("## Method\n\n")
            f.write(
                "For each comparison, the per-seed difference is AUC(model A) − AUC(model B). "
                "We used a two-sided Nadeau–Bengio corrected resampled t-test with the repeated-holdout "
                f"variance factor 1/n + n_test/n_train = {correction:.6f}, t degrees of freedom {len(seeds)-1}. "
                f"The five p-values were adjusted together with Holm's step-down method (family-wise alpha={alpha}). "
                "The confidence intervals below are corrected 95% intervals and are nominal per comparison, "
                "not simultaneous family-wise intervals.\n\n"
            )
            f.write("## Results\n\n")
            f.write("| Comparison (A − B) | Mean ΔAUC | Corrected 95% CI | Holm-adjusted p | A higher / tied / B higher |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for _, row in results.iterrows():
                f.write(
                    f"| {row.comparison} | {row.mean_auc_difference_a_minus_b:+.4f} | "
                    f"[{row.corrected_ci_lower:+.4f}, {row.corrected_ci_upper:+.4f}] | "
                    f"{row.p_value_holm:.4g} | "
                    f"{row.seeds_a_higher}/{row.seeds_tied}/{row.seeds_b_higher} |\n"
                )
            f.write("\n## Interpretation\n\n")
            f.write(f"- **WGAN augmentation:** {wgan_conclusion}\n")
            f.write(f"- **TabPFN vs. paper-config GB:** {model_conclusion}\n\n")
            f.write(
                "These tests compare repeated splits from one cohort. They do not establish performance "
                "on an independent hospital or patient population. A non-significant result is not proof "
                "of equivalence or of a zero augmentation effect.\n"
            )

        print(f"Experiment 12 | {len(seeds)} matched splits | Nadeau–Bengio corrected paired t tests")
        print(results[["comparison", "mean_auc_difference_a_minus_b", "p_value_holm"]].to_string(index=False))
        print(f"\nSaved report: {report_path}")
        return {"results": results, "paired_differences": deltas_for_export}


def main():
    Experiment12()()


if __name__ == "__main__":
    main()
