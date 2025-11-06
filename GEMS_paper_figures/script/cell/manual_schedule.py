import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────── Configuration ──────────────── #

LABEL_FONT_OPTIONS = {'weight': 'bold', 'size': 10, 'family': 'Helvetica'}
AX_X_FONTSIZE    = 10
AX_Y_FONTSIZE    = 10
LEGEND_FONT_SIZE = 10
XAXIS_LABEL_NUM  = 7

WIDTH = 560/72  # Width in inches
HEIGHT = 230/72  # Height in inches

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

# ──────────────────────────────────────────────── #

# Set label as "iPS" for hiPSC and "HEK293A" for HEK
def get_cell_label(cell_type):
    """細胞タイプに基づいてラベルを返す"""
    if cell_type == "hiPSC":
        return "hiPSC"
    elif cell_type == "HEK":
        return "HEK293A"
    else:
        return cell_type

# SAVE_PATH = "figure/cell_manual_schedule.png"
SAVE_PATH = "figure/Fig5-D.png"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# データ読み込み
df = pd.read_csv(
    "data/241106_CCDS_253g1-hek293a_report/experimental_schedule.tsv",
    sep="\t",
    dayfirst=True
)
df["Carrying in"]        = pd.to_datetime(df["Carrying in"], dayfirst=True)
df["Carrying out"]       = pd.to_datetime(df["Carrying out"], dayfirst=True)
df["Abnormal finishing"] = pd.to_datetime(df["Abnormal finishing"], dayfirst=True)

# 細胞タイプに基づいてラベルを設定
df["Cell label"] = df["Cell type"].apply(get_cell_label)

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

# 手動入力期間のハイライト
manual_start = pd.to_datetime("06/12/2024", dayfirst=True)
manual_end   = pd.to_datetime("10/12/2024", dayfirst=True)
ax.axvspan(manual_start, manual_end, color="gray", alpha=0.3, zorder=0)
ax.text(
    (manual_start + (manual_end - manual_start) / 2),
    len(df) + 0.5,
    "Manual experimental result input",
    ha="center", va="bottom",
    fontsize=AX_Y_FONTSIZE, color="gray"
)

# ガントバー（細胞タイプに基づいて色分け）
colors = {"hiPSC": "#1f77b4", "HEK293A": "#ff7f0e"}  # 青とオレンジ
legend_added = set()  # 凡例に追加済みのラベルを追跡

for _, row in df.iterrows():
    start = row["Carrying in"]
    end   = row["Carrying out"] if pd.notnull(row["Carrying out"]) else row["Abnormal finishing"]
    duration = (end - start).days
    cell_label = row["Cell label"]
    color = colors.get(cell_label, "gray")
    
    # 凡例用のラベル（初回のみ）
    bar_label = cell_label if cell_label not in legend_added else ""
    if cell_label not in legend_added:
        legend_added.add(cell_label)
    
    ax.barh(
        row["Cell id"], duration, left=start, 
        alpha=0.7, zorder=1, color=color, label=bar_label
    )

# 異常終了マーカー
ab_df = df[df["Finishing type"] == "Abnormal"]
ax.scatter(
    ab_df["Abnormal finishing"],
    ab_df["Cell id"],
    marker="x", s=70, color="red", zorder=2,
    label="Abnormal finishing"
)

# 軸ラベル
ax.set_xlabel("Date", **LABEL_FONT_OPTIONS)
ax.set_ylabel("Cell ID", **LABEL_FONT_OPTIONS)

# Y軸のティックとラベルを細胞タイプで置換
cell_ids = df["Cell id"].tolist()
ytick_labels = [f"{row['Cell id'].replace('hiPSC', 'hiPSC').replace('HEK', 'HEK293A')}" for _, row in df.iterrows()]
ax.set_yticks(range(len(cell_ids)))
ax.set_yticklabels(ytick_labels)

# 日付フォーマット指定
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b/%Y"))
fig.autofmt_xdate()

# 凡例
ax.legend(loc="upper right")


plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.close(fig)
