import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def create_gantt_chart(csv_path, save_path=None):
    """
    指定したCSVファイルからデータを読み込み、ガントチャートを作成する関数。
    
    Parameters:
        csv_path (str): CSVファイルのパス
        save_path (str, optional): 保存先のパス。指定しない場合は画面に表示する。
    """
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    
    # 日時データをdatetime型に変換
    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    df['scheduled_time_end'] = df['scheduled_time'] + pd.to_timedelta(df['processing_time_min'], unit='m')
    
    # ガントチャートの準備
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # タスクのプロット
    for i, row in df.iterrows():
        start_time = row['scheduled_time']
        end_time = row['scheduled_time_end']
        
        # タスクをガントチャートに追加
        ax.barh(
            y=row['experiment_operation'], 
            width=(end_time - start_time), 
            left=start_time,
            height=0.4, align='center', label=row['experiment_operation']
        )

    # X軸を日時フォーマットに設定
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # 軸のラベルとタイトルを設定
    plt.xlabel('Time')
    plt.ylabel('Operation')
    plt.title('Gantt Chart of Operations')

    # 凡例を表示
    # ax.legend(loc='upper right')

    # ガントチャートを保存または表示
    if save_path:
        plt.savefig(save_path)
        print(f"ガントチャートを {save_path} に保存しました。")
    else:
        plt.show()