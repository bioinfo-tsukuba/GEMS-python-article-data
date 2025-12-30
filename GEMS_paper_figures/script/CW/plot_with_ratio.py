import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TITLE_FONT_SIZE  = 18
LABEL_FONT_OPTIONS = {'weight': 'bold', 'size': 10, 'family': 'Helvetica'}
AX_X_FONTSIZE    = 10
AX_Y_FONTSIZE    = 10
LEGEND_FONT_SIZE = 10

WIDTH = 320/72  # Width in inches
HEIGHT = 360/72  # Height in inches

plt.rcParams.update({
    'figure.figsize': (WIDTH, HEIGHT),
    'font.size': AX_X_FONTSIZE,
    'axes.titlesize': LABEL_FONT_OPTIONS['size'],
    'axes.labelsize': AX_Y_FONTSIZE,
    'xtick.labelsize': AX_X_FONTSIZE,
    'ytick.labelsize': AX_Y_FONTSIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'axes.titleweight': LABEL_FONT_OPTIONS['weight'],
    'axes.labelweight': LABEL_FONT_OPTIONS['weight'],
    'font.family': 'Helvetica',
})



# ファイル読み込み（パスは適宜変更）
file_path = "artifacts/CW/filtered_with_round_scored.csv"
df_filtered = pl.read_csv(file_path).filter(pl.col("tag") == "Sample")


# ROUND と row ごとにスコアの平均、各Colour_ratioの値（同一グループ内の最初の値）を取得し、ラベルを作成
grouped = (
    df_filtered.group_by(["ROUND", "row"])
    .agg(
         pl.col("Score").mean().alias("Score"),
         pl.col("Color1_ratio").first().alias("Color1_ratio"),
         pl.col("Color2_ratio").first().alias("Color2_ratio"),
         pl.col("Color3_ratio").first().alias("Color3_ratio")
    )
    .with_columns(
         (
            #  pl.col("ROUND").cast(pl.Int64).cast(pl.Utf8) + "-" + 
            pl.col("row")).alias("label"
                                 )
    )
    .sort(by=["ROUND", "Score"])
)

# print(grouped.head())
# print all without skipping
for row in grouped.iter_rows():
    print(row)

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
xmax = 200

# 各ROUNDにつき、左：Color比率（積み上げ棒）、右：Scoreランキングの横棒グラフ
fig, axes = plt.subplots(
    nrows=n_rounds, 
    ncols=2, 
    figsize=(WIDTH, HEIGHT), 
    sharey='row', 
    sharex='col', 
    constrained_layout=True
)

if n_rounds == 1:
    axes = [axes]

# 統一された色の指定（例：Color1=青、Color2=オレンジ、Color3=緑）
color1 = "#CC4125"
color2 = "#1155CC"
color3 = "#F1C232"

legend_handles = [
    Patch(facecolor=color1, edgecolor="black", label="Acidic red solution"),
    Patch(facecolor=color2, edgecolor="black", label="Basic clear solution"),
    Patch(facecolor=color3, edgecolor="black", label="BTB"),
]

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
    
    ax_left.set_title(f"ROUND {int(round_id)} - Volume Fractions")
    ax_left.set_xlabel("Percentage", **LABEL_FONT_OPTIONS)
    ax_left.set_xlim(0, 100)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(subdata["label"].to_list())
    ax_left.grid(axis='x', linestyle='--', alpha=0.7)
    # ax_left.legend(["Colour1", "Colour2", "Colour3"], loc='lower right')
    
    # 右側：Scoreランキングの横棒グラフ
    ax_right = axes[idx][1]
    scores = subdata["Score"].to_list()
    ax_right.barh(y, scores, color="gray", edgecolor='black')
    
    # 各ラウンドの最小スコア（最良スコア）に点線を追加
    min_score = min(scores)
    ax_right.axvline(x=min_score, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Min: {min_score:.1f}')
    
    ax_right.set_title(f"ROUND {int(round_id)} - Score Ranking")
    ax_right.set_xlabel("Avg Score")
    ax_right.set_xlim(0, xmax)
    # Delete the y-ticks to avoid overlap
    ax_right.set_yticks([])
    ax_right.set_yticklabels([])
    ax_right.grid(axis='x', linestyle='--', alpha=0.7)
    ax_right.legend(loc='lower right', fontsize=8)

for ax_row in axes:
    for ax in ax_row:
        ax.label_outer()


# plt.legend(handlelength=1)
fig.legend(
    handles=legend_handles,
    loc="upper left",
    ncols=3,
    frameon=False,
    bbox_to_anchor=(0.015, 1.00),
    fontsize=LEGEND_FONT_SIZE,
    handlelength=1.2
)


# plt.suptitle("Score Ranking and Colour Ratio Composition by ROUND", fontsize=TITLE_FONT_SIZE)
fig.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("script/CW/score_ranking_with_color_ratio.png", dpi=300)
SAVE_PATH = "figure/Fig4-F.png"
plt.savefig(SAVE_PATH, dpi=300)
plt.show()
