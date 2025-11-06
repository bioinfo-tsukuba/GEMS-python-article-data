import polars as pl

# CSV読み込み
df = pl.read_csv("dry_analyse/filtered_with_round_scored.csv")

# 必要な列だけ抽出
df = df.select(["ROUND", "Color1_ratio", "Color2_ratio", "Color3_ratio", "Score"])

# Ratioごとに平均Scoreを計算
grouped = (
    df.group_by(["ROUND", "Color1_ratio", "Color2_ratio", "Color3_ratio"])
      .agg(pl.col("Score").mean().alias("mean_Score"))
)

# 各ROUNDごとに最小の平均Scoreを持つRatioを抽出
best_ratios = (
    grouped.sort("mean_Score")
           .group_by("ROUND")
           .first()
)

print(best_ratios)