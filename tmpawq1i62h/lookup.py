import polars as pl

teams = pl.read_csv("data/MTeams.csv").select("TeamID", "TeamName")
seeds = pl.read_csv("data/MNCAATourneySeeds.csv").filter(pl.col("Season") == 2026)
slots = pl.read_csv("data/MNCAATourneySlots.csv").filter(pl.col("Season") == 2026)

tid_to_name = dict(zip(teams["TeamID"].to_list(), teams["TeamName"].to_list()))
seed_to_team = {}
for r in seeds.iter_rows(named=True):
    seed_to_team[r["Seed"]] = r["TeamID"]

# All teams in each region
for region in ["W", "X", "Y", "Z"]:
    rnames = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
    print(f"\n=== {rnames[region]} REGION ===")
    for seed_str, tid in sorted(seed_to_team.items()):
        if seed_str[0] == region:
            print(f"  {seed_str}: {tid_to_name.get(tid, '?')} (TID={tid})")

print("\n=== ALL SLOTS (South region Y) ===")
for r in slots.sort("Slot").iter_rows(named=True):
    s = r["Slot"]
    if "Y" in s or s.startswith("R5") or s.startswith("R6"):
        print(f"  {s}: {r['StrongSeed']} vs {r['WeakSeed']}")
