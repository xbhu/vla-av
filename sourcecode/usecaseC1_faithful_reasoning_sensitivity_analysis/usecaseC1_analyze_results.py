"""
Reads the CSV produced by usecaseC1_run_battery.py and runs a Wilcoxon signed-rank test.
"""

import pandas as pd
from scipy.stats import wilcoxon

CSV_PATH = "./outputs/usecaseC1/c1_battery_results.csv"

def report(name, values):
    values = pd.Series(values).dropna()
    if len(values) < 3:
        print(f"{name}: too few samples (n={len(values)}), skipping test")
        return
    stat, p = wilcoxon(values - 1e-9)
    print(f"{name}: n={len(values)}, mean={values.mean():.3f}m, median={values.median():.3f}m, "
          f"p95={values.quantile(0.95):.3f}m, Wilcoxon p={p:.4g}")

def main():
    df = pd.read_csv(CSV_PATH)

    print("=== Experiment 0: Noise floor ===")
    print(df["endpoint_l2_noise_floor"].describe())

    print("\n=== Experiment 1: CoT on/off ===")
    report("endpoint_l2_cot_off", df["endpoint_l2_cot_off"])
    report("mean_l2_cot_off", df["mean_l2_cot_off"])

    print("\n=== Experiment 2: Instruction counterfactual ===")
    report("endpoint_l2_instr_cf1", df["endpoint_l2_instr_cf1"])
    report("endpoint_l2_instr_cf2", df["endpoint_l2_instr_cf2"])

    print("\n=== Experiment 3: Perception masking ===")
    report("endpoint_l2_perception_mask", df["endpoint_l2_perception_mask"])

    noise_median = df["endpoint_l2_noise_floor"].median()
    print(f"\nNoise floor median: {noise_median:.3f}m")

if __name__ == "__main__":
    main()
