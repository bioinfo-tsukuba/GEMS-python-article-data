from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates

# データ読み込みと前処理
artifact_path = Path("data/241106_CCDS_253g1-hek293a_report/processed_data")
time_converted_task_path = artifact_path / "time_converted_task.csv"

schedule_df = pd.read_csv(time_converted_task_path)
# schedule_df = schedule_df[:10]

# datetime 型に変換
schedule_df['scheduled_time'] = pd.to_datetime(schedule_df['scheduled_time'])
schedule_df['end_time'] = schedule_df['scheduled_time'] + pd.to_timedelta(schedule_df['processing_time_min'], unit='m')
schedule_df['optimal_time_reference_time'] = pd.to_datetime(schedule_df['optimal_time_reference_time'])
schedule_df['estimated_start_time'] = schedule_df['optimal_time_reference_time'] - pd.to_timedelta(schedule_df['processing_time_min'], unit='m')

# 実験ごとのリストを取得し、サブプロット設定
experiments = schedule_df['experiment_name'].unique()
n_experiments = len(experiments)
total_rows = n_experiments

fig, axes = plt.subplots(nrows=total_rows, ncols=1, figsize=(20, total_rows*5), sharex=True)
# 各実験ごとにガントチャートを描画
for i, exp in enumerate(experiments):
    exp_df = schedule_df[schedule_df['experiment_name'] == exp]
    
    # ラベルを生成してソート
    exp_df['label'] = exp_df.apply(lambda row: f"{row['experiment_uuid']}-{row['experiment_operation']}" if row['experiment_operation'] != 'LabwareRefill' else row['experiment_operation'], axis=1)
    # iPS_Experiment_ を IE に置換
    exp_df['label'] = exp_df['label'].str.replace('iPS_Experiment_', 'IE')
    # HEK_Experiment_ を HE に置換
    exp_df['label'] = exp_df['label'].str.replace('HEK_Experiment_', 'HE')
    exp_df['label'] = exp_df['label'].str.replace('iPS', '').str.replace('HEK', '')
    exp_df = exp_df.sort_values(by='label')  # ラベルでソート
    
    ax_sched = axes[i]     # 上段: scheduled_time 用

    # 上段: scheduled_time に基づくガントチャート
    for idx, row in exp_df.iterrows():
        start_time = mdates.date2num(row['scheduled_time'])
        end_time = mdates.date2num(row['end_time'])
        duration = end_time - start_time

        ax_sched.barh(
            row['label'],  # ソートされたラベルを使用
            duration,
            left=start_time,
            color='skyblue',
            edgecolor='black'
        )
        ax_sched.text(
            start_time + duration / 2,
            row['label'],  # ラベル位置
            "",
            ha='center',
            va='center',
            fontsize=8,
            clip_on=True  # 画面外テキストをクリップ
        )
    ax_sched.tick_params(axis='y', which='major', pad=100)  # ラベルの位置を調整
    ax_sched.set_yticklabels(ax_sched.get_yticklabels(), ha='left')  # ラベルを左揃え

    ax_sched.set_ylabel('Scheduled')
    ax_sched.set_title(f'Experiment: {exp} - Scheduled Time')
# 共通のX軸フォーマット設定
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=45)

axes[-1].set_xlabel('Time')

plt.tight_layout()

# アニメーションの設定
# 全データにおける最小値・最大値を取得
all_scheduled_min = mdates.date2num(schedule_df['scheduled_time'].min())
all_end_max = mdates.date2num(schedule_df['end_time'].max())
window_width = (all_end_max - all_scheduled_min) / 4  # 表示ウィンドウの幅

def update(frame):
    # 各フレームごとにX軸の範囲を更新
    new_left = all_scheduled_min + (all_end_max - all_scheduled_min - window_width) * frame / 100
    for ax in axes:
        ax.set_xlim(new_left, new_left + window_width)
    return axes

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

# アニメーションを保存
# ani.save('gantt_scroll_executed_experiments.mp4', writer='ffmpeg')
save_path = 'figure/gantt_scroll_executed_experiments.gif'
if not os.path.exists('figure'):
    os.makedirs('figure')
ani.save(save_path, writer='pillow', fps=10, dpi=100)
