import argparse
import numpy as np
import pandas as pd

# settings
G_MIN = 0.12   # lowest number of the green light for each side
EPS   = 1e-9   # if the delay is zero(so we cant divided by 0)


# load and clean the file
def load_and_clean(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"timestamp", "link_name", "travel_time_seconds", "traffic_delay_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"columns not exist: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["travel_time_seconds", "traffic_delay_seconds"])

    # filtering the data if it is unrealistic
    n_before = len(df)
    df = df[(df["travel_time_seconds"] > df["traffic_delay_seconds"])
            & (df["travel_time_seconds"] > 0)].copy()
    n_removed = n_before - len(df)
    if n_removed:
        print(f"delete {n_removed}. Invalid row")

    # link
    df["artery"] = df["link_name"].str.extract(r"ZoneA_([A-Za-z]+)_")
    if df["artery"].isna().any():
        bad = df.loc[df["artery"].isna(), "link_name"].unique()
        raise ValueError(f"The name of the artery could not be retrieved from: {bad}")

    return df


# congestion= delay time/ free travel time
def compute_arterial_congestion(df: pd.DataFrame) -> pd.DataFrame:

    free_time = df["travel_time_seconds"] - df["traffic_delay_seconds"]
    df["congestion"] = df["traffic_delay_seconds"] / free_time

    arterial = (df.groupby(["timestamp", "artery"])["congestion"]
                  .mean()
                  .reset_index())

    matrix = arterial.pivot(index="timestamp", columns="artery", values="congestion")

    matrix = matrix.dropna()
    return matrix


def demand_shares(congestion: np.ndarray) -> np.ndarray:

    total = congestion.sum()
    n = len(congestion)
    if total < EPS:                       # Night/smooth flow: no congestion to redistribute
        return np.full(n, 1.0 / n)
    return congestion / total


def allocate_green(demand: np.ndarray, g_min: float = G_MIN) -> np.ndarray:

    g = demand.astype(float).copy()
    fixed = np.zeros(len(demand), dtype=bool)

    for _ in range(len(demand)):
        # Making sure that all the green lights are above or equal the minimum 12%
        below = (g < g_min) & (~fixed)
        if not below.any():
            break
        g[below] = g_min
        fixed[below] = True
        # re setting the weights to be equal to 1.0 or 100%
        free = ~fixed
        remaining = 1.0 - g[fixed].sum()
        if free.any() and demand[free].sum() > 0:
            g[free] = remaining * demand[free] / demand[free].sum()
        else:
            break
    return g


def delay_index(demand: np.ndarray, green: np.ndarray) -> float:
    # Demand²/the green line= the delay
    return float(np.sum(demand**2 / green))


# ----------------------------- 4) Execution on all snapshots -----------------------------
def run_optimization(matrix: pd.DataFrame, g_min: float = G_MIN) -> pd.DataFrame:
    """Applies the algorithm to each snapshot and compares equal allocation versus SCATS allocation."""
    arteries = list(matrix.columns)
    equal = np.full(len(arteries), 1.0 / len(arteries))   # Naive baseline

    records = []
    for ts, row in matrix.iterrows():
        d = demand_shares(row.values)
        g = allocate_green(d, g_min)
        rec = {
            "timestamp":    ts,
            "delay_before": delay_index(d, equal),
            "delay_after":  delay_index(d, g),
        }
        for i, a in enumerate(arteries):
            rec[f"demand_{a}"] = d[i]
            rec[f"green_{a}"]  = g[i]
        records.append(rec)

    res = pd.DataFrame(records)
    res["improvement_pct"] = (
        (res["delay_before"] - res["delay_after"]) / res["delay_before"] * 100
    )
    return res


# ----------------------------- 5) Results Summary -----------------------------
def summarize(res: pd.DataFrame) -> None:
    tot_b = res["delay_before"].sum()
    tot_a = res["delay_after"].sum()
    print("\n================= Results Summary =================")
    print(f"Processed snapshots count : {len(res)}")
    print(f"Total delay (Before)      : {tot_b:.1f}")
    print(f"Total delay (After)       : {tot_a:.1f}")
    print(f"Overall reduction         : {(tot_b - tot_a) / tot_b * 100:.1f}%")

    # Detail by period — the gain is related to the asymmetry between arteries
    hour = res["timestamp"].dt.hour
    def bucket(h):
        if 7 <= h <= 9:   return "Morning Peak (7-9)"
        if 16 <= h <= 20: return "Evening Peak (16-20)"
        if 0 <= h <= 5:   return "Night (0-5)"
        return "Off-Peak"
    res = res.assign(period=hour.map(bucket))
    by = (res.groupby("period")["improvement_pct"]
             .agg(["size", "mean"]).round(1)
             .rename(columns={"size": "snapshot_count", "mean": "avg_improvement_pct"}))
    print("\n--- Average Improvement by Period ---")
    print(by.to_string())
    print("=================================================\n")


# ----------------------------- Entry Point -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: SCATS Algorithm for Green Time Allocation."
    )
    parser.add_argument("--input",  default="riyadh_traffic_links.csv",
                        help="Path to the traffic data file (CSV).")
    parser.add_argument("--output", default="scats_results.csv",
                        help="Path to save the algorithm results (CSV).")
    parser.add_argument("--g-min",  type=float, default=G_MIN,
                        help="Minimum green split for each direction (default 0.12).")
    args = parser.parse_args()

    df       = load_and_clean(args.input)
    matrix   = compute_arterial_congestion(df)
    print(f"[INFO] Arteries: {list(matrix.columns)} | Completed snapshots: {len(matrix)}")

    results  = run_optimization(matrix, g_min=args.g_min)
    summarize(results)

    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Complete results ({len(results)} rows) saved to: {args.output}")


if __name__ == "__main__":
    main()