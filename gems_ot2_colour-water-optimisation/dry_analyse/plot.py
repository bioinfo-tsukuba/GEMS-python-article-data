import polars as pl
import matplotlib.pyplot as plt

# ファイル読み込み（パスは適宜変更）
file_path = "dry_analyse/filtered_with_round.csv"
df_filtered = pl.read_csv(file_path)

# ターゲットRGB（固定値として与えられていたもの）
target_rgb = {'R': 51, 'G': 76.0, 'B': 127}

# スコア計算（マンハッタン距離）
df_filtered = df_filtered.with_columns(
    (
        (pl.col("R") - target_rgb["R"]).abs() +
        (pl.col("G") - target_rgb["G"]).abs() +
        (pl.col("B") - target_rgb["B"]).abs()
    ).alias("Score")
)

# ROUND と row ごとにスコアを平均化し、ラベル列を追加,roundでソート→scoreでソート
grouped = (
    df_filtered.group_by(["ROUND", "row"])
    .agg(pl.col("Score").mean().alias("Score"))
    .with_columns(
        (pl.col("ROUND").cast(pl.Int64).cast(pl.Utf8) + "-" + pl.col("row")).alias("label")
    )
    .sort(by=["ROUND", "Score"])
)


print(grouped.head())

# ROUND ごとにグループ分けしてプロット
unique_rounds = sorted(grouped["ROUND"].unique().to_list())
n_rounds = len(unique_rounds)

# スコアの最大値を取得し、すべてのプロットの x 軸を統一
xmax = 255*3

# プロットの作成
fig, axes = plt.subplots(nrows=1, ncols=n_rounds, figsize=(4 * n_rounds, 6), sharey=True)

# サブプロットが1つの場合の処理
if n_rounds == 1:
    axes = [axes]

# Score を ROUND ごとにランク付け
grouped = (
    grouped.with_columns(
        pl.col("Score").rank("ordinal").over("ROUND").alias("Score_rank")
    )
)

# プロットの作成（Y軸を Score_rank に）
fig, axes = plt.subplots(nrows=1, ncols=n_rounds, figsize=(4 * n_rounds, 6), sharey=True)

if n_rounds == 1:
    axes = [axes]

for ax, round_id in zip(axes, unique_rounds):
    subdata = grouped.filter(pl.col("ROUND") == round_id)
    ax.barh(subdata["Score_rank"].to_list(), subdata["Score"].to_list())
    ax.set_title(f"ROUND {int(round_id)}")
    ax.set_xlabel("Avg Score")
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.set_yticks(subdata["Score_rank"].to_list())
    ax.set_yticklabels(subdata["label"].to_list())
    ax.grid(axis='x')

axes[0].set_ylabel("Score Rank (lower is better)")
plt.suptitle("Score Ranking by ROUND", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("dry_analyse/score_ranking_by_round.png")