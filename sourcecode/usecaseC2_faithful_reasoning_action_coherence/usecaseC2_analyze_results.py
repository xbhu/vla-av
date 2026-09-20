"""
Read the CSV produced by usecaseC2_run_consistency.py, report lateral/
longitudinal agreement rates, a confusion matrix (stated category vs.
what the trajectory actually does), and the two new N/A categories
(no explicit decision stated at all; lateral N/A because the stated
action is "stop").
"""

import pandas as pd

CSV_PATH = "./outputs/usecaseC2/c2_consistency_results.csv"


def report_agreement(df, col_match, name):
    scored = df[col_match].notna()
    n_total = len(df)
    n_scored = scored.sum()
    n_match = df.loc[scored, col_match].sum()

    print(f"\n=== {name} ===")
    print(f"Total scenes: {n_total}")
    print(f"Scenes with a scoreable comparison: {n_scored} ({n_scored/n_total*100:.1f}%)")
    if n_scored > 0:
        print(f"Agreement rate: {n_match}/{n_scored} = {n_match/n_scored*100:.1f}%")
    else:
        print("No comparable samples.")


def report_confusion(df, col_stated, col_traj, name):
    parseable = df[col_stated].notna() & df[col_traj].notna()
    n_parseable = parseable.sum()
    print(f"\n--- {name} confusion matrix (rows = stated, cols = trajectory) ---")
    if n_parseable == 0:
        print("No comparable samples.")
        return
    confusion = pd.crosstab(
        df.loc[parseable, col_stated], df.loc[parseable, col_traj],
        rownames=["stated"], colnames=["trajectory"]
    )
    print(confusion)


def main():
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)

    print(f"CSV row count: {n_total}")

    no_decision_rate = df["no_decision_stated"].mean() * 100
    print(f"Scenes with NO explicit 'Best Driving Action' statement at all: "
          f"{df['no_decision_stated'].sum()}/{n_total} ({no_decision_rate:.1f}%)")
    print("(these are excluded from all agreement stats below -- there is nothing to compare)")

    na_stop_rate = df["lateral_is_na_due_to_stop"].mean() * 100
    print(f"Scenes where lateral is N/A because the stated action is just 'stop': "
          f"{df['lateral_is_na_due_to_stop'].sum()}/{n_total} ({na_stop_rate:.1f}%)")
    print("(excluded from lateral agreement, but STILL included in longitudinal agreement)")

    report_agreement(df, "lateral_match", "Lateral action agreement")
    report_confusion(df, "stated_lateral", "traj_lateral", "Lateral")

    report_agreement(df, "longitudinal_match", "Longitudinal action agreement")
    report_confusion(df, "stated_longitudinal", "traj_longitudinal", "Longitudinal")

    both = df["both_match"].dropna()
    if len(both) > 0:
        print(f"\n=== Lateral AND longitudinal both agree ===")
        print(f"{int(both.sum())}/{len(both)} = {both.mean()*100:.1f}%")

    # Directional check: among longitudinal mismatches, is there a
    # systematic bias (e.g. model states "stop"/"decelerate" but the
    # trajectory actually accelerates)?
    mismatches = df[(df["longitudinal_match"] == False)]
    if len(mismatches) > 0:
        print("\n=== Longitudinal mismatch pairs (stated -> actual), to check for a systematic direction ===")
        print(mismatches.groupby(["stated_longitudinal", "traj_longitudinal"]).size()
              .sort_values(ascending=False))


if __name__ == "__main__":
    main()
