import polars as pl

sample = pl.read_csv("data/SampleSubmissionStage2.csv")
sub = pl.read_csv("artifacts/latest/submission.csv")

print(f"Sample: {sample.height} rows")
print(f"Submission: {sub.height} rows")

sample_ids = set(sample["ID"].to_list())
sub_ids = set(sub["ID"].to_list())
extra = sub_ids - sample_ids
missing = sample_ids - sub_ids
print(f"Extra IDs in submission: {len(extra)}")
print(f"Missing IDs from submission: {len(missing)}")
if extra:
    for e in sorted(extra)[:10]:
        print(f"  EXTRA: {e}")
if missing:
    for m in sorted(missing)[:10]:
        print(f"  MISSING: {m}")

pred = sub["Pred"]
print(f"Pred min: {pred.min()}")
print(f"Pred max: {pred.max()}")
print(f"Pred nulls: {pred.null_count()}")
print(f"Pred NaN: {pred.is_nan().sum()}")

# Check ID format
bad_format = sub.filter(~pl.col("ID").str.contains(r"^2026_\d+_\d+$"))
print(f"Bad ID format rows: {bad_format.height}")

# Check IDs are ordered correctly (low < high)
parsed = sub.with_columns(
    pl.col("ID").str.split("_").list.get(1).cast(pl.Int64).alias("t1"),
    pl.col("ID").str.split("_").list.get(2).cast(pl.Int64).alias("t2"),
)
wrong_order = parsed.filter(pl.col("t1") >= pl.col("t2"))
print(f"Wrong order IDs (t1 >= t2): {wrong_order.height}")

# Check for duplicates
dupes = sub.group_by("ID").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print(f"Duplicate IDs: {dupes.height}")
