import polars as pl

for fname, champ_tid, champ_name, f4 in [
    ("submission_duke_title.csv", 1181, "Duke", {1112, 1181, 1196, 1235}),
    ("submission_houston_title.csv", 1211, "Houston", {1112, 1181, 1196, 1211}),
]:
    sub = pl.read_csv(f"artifacts/latest/{fname}")
    bad = sub.filter(~pl.col("ID").str.contains(r"^\d{4}_\d+_\d+$")).height
    null_p = sub.filter(pl.col("Pred").is_null()).height
    oor = sub.filter((pl.col("Pred") < 0) | (pl.col("Pred") > 1)).height
    print(f"\n=== {fname} ===")
    print(f"Rows: {sub.height:,}  Bad IDs: {bad}  Nulls: {null_p}  OOR: {oor}")
    print(f"Pred range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")

    # Check champion's win probs vs other F4 teams
    other_f4 = sorted(f4 - {champ_tid})
    print(f"\n{champ_name} (TID {champ_tid}) vs other F4 teams:")
    for opp in other_f4:
        low, high = min(champ_tid, opp), max(champ_tid, opp)
        row = sub.filter(pl.col("ID") == f"2026_{low}_{high}")
        if row.height:
            pred = row["Pred"][0]
            if champ_tid == low:
                p_champ = pred
            else:
                p_champ = 1 - pred
            print(f"  vs TID {opp}: P({champ_name} wins) = {p_champ:.4f}")
