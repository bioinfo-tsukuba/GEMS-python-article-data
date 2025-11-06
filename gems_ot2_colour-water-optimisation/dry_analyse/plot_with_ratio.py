import polars as pl
import matplotlib.pyplot as plt

# ファイル読み込み（パスは適宜変更）
file_path = "dry_analyse/filtered_with_round_scored.csv"
df_filtered = pl.read_csv(file_path).filter(pl.col("tag") == "Sample")


# ROUND と row ごとにスコアの平均、各Color_ratioの値（同一グループ内の最初の値）を取得し、ラベルを作成
grouped = (
    df_filtered.group_by(["ROUND", "row"])
    .agg(
         pl.col("Score").mean().alias("Score"),
         pl.col("Color1_ratio").first().alias("Color1_ratio"),
         pl.col("Color2_ratio").first().alias("Color2_ratio"),
         pl.col("Color3_ratio").first().alias("Color3_ratio")
    )
    .with_columns(
         (pl.col("ROUND").cast(pl.Int64).cast(pl.Utf8) + "-" + pl.col("row")).alias("label")
    )
    .sort(by=["ROUND", "Score"])
)

# PRINT ALL ROW's Color1_ratio, Color2_ratio, Color3_ratio, Score
it = 0
for row in grouped.iter_rows(named=True):
    if it % 8 == 0:
        Color1_ratio = row["Color1_ratio"]
        Color2_ratio = row["Color2_ratio"]
        Color3_ratio = row["Color3_ratio"]
        Score = row["Score"]
        print(f"(ratio: {Color1_ratio:.2f}, {Color2_ratio:.2f}, {Color3_ratio:.2f}, Score: {Score:.2f})")
    it += 1

# 各ROUND内でScoreの順位を算出（順位が低いほどスコアが低い＝良い）
grouped = grouped.with_columns(
    pl.col("Score").rank("ordinal").over("ROUND").alias("Score_rank")
)

# 各Color_ratioの合計を求め、各値を100に正規化（割合に換算）
grouped = grouped.with_columns(
    (pl.col("Color1_ratio") + pl.col("Color2_ratio") + pl.col("Color3_ratio")).alias("total_ratio")
).with_columns(
    (pl.col("Color1_ratio") / pl.col("total_ratio") * 100).alias("perc1"),
    (pl.col("Color2_ratio") / pl.col("total_ratio") * 100).alias("perc2"),
    (pl.col("Color3_ratio") / pl.col("total_ratio") * 100).alias("perc3")
)

# ROUNDごとのユニークな値を取得
unique_rounds = sorted(grouped["ROUND"].unique().to_list())
n_rounds = len(unique_rounds)

# Score用のx軸最大値（マンハッタン距離の最大値）
# xmax = 255 * 3
# Scoreの最大値を自動取得
xmax = (grouped["Score"].max() // 10 + 1) * 10  # 10の倍数に切り上げ

magn = 0.6
# 各ROUNDにつき、左：Color比率（積み上げ棒）、右：Scoreランキングの横棒グラフ
fig, axes = plt.subplots(nrows=n_rounds, ncols=2, figsize=(12*magn, 3.6 * n_rounds*magn), sharey=True)

if n_rounds == 1:
    axes = [axes]

# 統一された色の指定（例：Color1=青、Color2=オレンジ、Color3=緑）
color1 = "#FF0000"
color2 = "#3333FF"
color3 = "#FFFF33"

for idx, round_id in enumerate(unique_rounds):
    subdata = grouped.filter(pl.col("ROUND") == round_id)
    y = subdata["Score_rank"].to_list()
    
    # 左側：積み上げ横棒グラフ（色比率）
    ax_left = axes[idx][0]
    perc1 = subdata["perc1"].to_list()
    perc2 = subdata["perc2"].to_list()
    perc3 = subdata["perc3"].to_list()
    
    # 各クエリごとに、まずColor1の部分
    ax_left.barh(y, perc1, color=color1, edgecolor='black')
    # Color2：先の長さ分だけ左にずらして重ねる
    left_second = [p for p in perc1]
    ax_left.barh(y, perc2, left=left_second, color=color2, edgecolor='black')
    # Color3：先2つの合計分だけ左にずらす
    left_third = [p1 + p2 for p1, p2 in zip(perc1, perc2)]
    ax_left.barh(y, perc3, left=left_third, color=color3, edgecolor='black')
    
    ax_left.set_title(f"Round {int(round_id)} - colour water ratio")
    ax_left.set_xlabel("Colour water ratio (%)")
    # x軸の範囲を0から100に設定
    ax_left.set_xlim(0, 100)
    # ax_left.set_yticks(y)
    # ax_left.set_yticklabels(subdata["label"].to_list())
    # y軸のラベルを非表示
    ax_left.set_yticklabels([])
    # ticksも非表示
    ax_left.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_left.grid(axis='x', linestyle='--', alpha=0.7)
    ax_left.legend(["Colour water 1", "Colour water 2", "Colour water 3"], loc='lower right', fontsize='small')

    # 右側：Scoreランキングの横棒グラフ
    ax_right = axes[idx][1]
    scores = subdata["Score"].to_list()
    ax_right.barh(y, scores, color="gray", edgecolor='black')
    ax_right.set_title(f"ROUND {int(round_id)} - Score ranking")
    ax_right.set_xlabel("Averaged Score")
    ax_right.set_xlim(0, xmax)
    ax_right.invert_yaxis()  # 良い順位（低い値）を上部に表示
    ax_right.grid(axis='x', linestyle='--', alpha=0.7)

# plt.suptitle("Score Ranking and Color Ratio Composition by ROUND", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("dry_analyse/score_ranking_with_color_ratio.png")
plt.show()


# # ─── ROUND ごとの代表値と改善率を計算 ───────────────────────────
# round_stats = (
#     grouped
#     .group_by("ROUND")
#     .agg(
#         pl.col("Score").mean().alias("mean_score"),   # ラウンド平均
#         pl.col("Score").min().alias("best_score")     # ラウンド最良（最小）
#     )
#     .sort("ROUND")
#     .with_columns(
#         # 平均スコア改善率（前回比：＋なら改善）
#         ((pl.col("mean_score").shift(1) - pl.col("mean_score"))
#          / pl.col("mean_score").shift(1) * 100).alias("mean_improve_pct"),
#         # 最良スコア改善率（前回比）
#         ((pl.col("best_score").shift(1) - pl.col("best_score"))
#          / pl.col("best_score").shift(1) * 100).alias("best_improve_pct")
#     )
# )

# print(round_stats)

# # ─── 改善率を算出した DataFrame round_stats は前段のまま ─────────────
# #  round_stats = [ ROUND, mean_score, best_score, mean_improve_pct, best_improve_pct ]

# # ─── 可視化：改善率を折れ線 + 数字で表示 ──────────────────────────
# fig, ax = plt.subplots(figsize=(8, 4))

# # 折れ線（平均改善率）
# ax.plot(round_stats["ROUND"], round_stats["mean_improve_pct"],
#         marker="o", label="Mean Improvement %", zorder=3)
# # 折れ線（最良値改善率）
# ax.plot(round_stats["ROUND"], round_stats["best_improve_pct"],
#         marker="s", label="Best Improvement %", zorder=3, linestyle="--")

# # ── 各点に改善率（小数 1 桁）を表示 ─────────────────────────────
# for x, y in zip(round_stats["ROUND"], round_stats["mean_improve_pct"]):
#     if y is not None:
#         ax.text(x, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

# for x, y in zip(round_stats["ROUND"], round_stats["best_improve_pct"]):
#     if y is not None:
#         ax.text(x, y, f"{y:.1f}%", ha="center", va="top", fontsize=8)

# # 軸設定
# ax.set_xlabel("Round")
# ax.set_ylabel("Improvement (%)")
# ax.set_title("Per-Round Improvement (Mean vs. Best)")
# ax.grid(True, linestyle="--", alpha=0.6)
# ax.legend()

# plt.tight_layout()
# plt.savefig("dry_analyse/round_improvement_lines.png")
# plt.show()

# # ─── 数値をコンソールにも一覧表示したい場合 ───────────────────────
# print(
#     round_stats
#     .select(["ROUND", "mean_improve_pct", "best_improve_pct"])
#     .with_columns([pl.col(c).round(2) for c in ["mean_improve_pct", "best_improve_pct"]])
# )