import polars as pl
sub = pl.read_csv("artifacts/latest/submission_isu_title.csv")
pairs = ["2026_1112_1181","2026_1112_1196","2026_1112_1235","2026_1181_1196","2026_1181_1235","2026_1196_1235"]
for p in pairs:
    row = sub.filter(pl.col("ID")==p)
    if row.height:
        print(f"{p}  Pred={row['Pred'][0]:.4f}")
print()
print("Iowa St win probs in F4/Championship:")
for p in ["2026_1112_1235","2026_1181_1235","2026_1196_1235"]:
    row = sub.filter(pl.col("ID")==p)
    pred = row["Pred"][0]
    opp = p.split("_")[1]
    print(f"  vs TID {opp}: P(Iowa St wins) = {1-pred:.4f}")
