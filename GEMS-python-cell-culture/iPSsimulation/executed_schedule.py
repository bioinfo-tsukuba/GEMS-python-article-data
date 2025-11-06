

from datetime import datetime
import json
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd




def get_executed_schedule(dir: Path, executed_schedule_csv: Path, time_unit: str = 'm') -> pd.DataFrame:
    """_summary_

    Args:
        dir (Path): _description_
        executed_schedule_csv (Path): _description_
        time_unit (str, optional): _description_. Defaults to 'm'.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """

    executed_schedule_df = pd.DataFrame()
    
    step_dirs = [
        d for d in dir.iterdir()
        if d.is_dir() and d.name.startswith('step_') and d.name[5:].isdigit()
    ]

    step_dirs.sort()

    for step_dir in step_dirs:
        print(step_dir)

        experiment_result_js = step_dir / "experiment_result.json"
        if not experiment_result_js.exists():
            print("No experiment_result.json")
            continue

        with experiment_result_js.open() as f:
            experiment_result = json.load(f)
            print(experiment_result)
        task_group_id: int = experiment_result['task_group_id']
        task_id: int = experiment_result['task_id']
        print(f"task_group_id: {task_group_id}, task_id: {task_id}")

        schedule_csv = step_dir / "schedule.csv"
        if not schedule_csv.exists():
            print("No schedule.csv")
            continue

        schedule_df = pd.read_csv(schedule_csv)

        # Get the row
        row = schedule_df[
            (schedule_df['task_group_id'] == task_group_id) &
            (schedule_df['task_id'] == task_id)
        ]
        print(row)
        if row.empty:
            print("No row")
            continue
        elif row["processing_time"][0] == None or row["processing_time"][0] == 0:
            print("No processing_time")
            continue
        elif len(row) > 1:
            print("Multiple rows")
            raise ValueError
            continue

        row['step'] = step_dir.name

        executed_schedule_df = pd.concat([executed_schedule_df, row])

    # Convert the scheduled_time to datetime following the time_unit
    executed_schedule_df['scheduled_time_datetime'] = pd.to_datetime(executed_schedule_df['scheduled_time'], unit=time_unit)

    local_timezone = datetime.now().astimezone().tzinfo
    print(f"Local timezone: {local_timezone}")

    # tz_localize
    executed_schedule_df['scheduled_time_datetime'] = executed_schedule_df['scheduled_time_datetime'].apply(lambda x: x.tz_localize('UTC').tz_convert(local_timezone))

    
    executed_schedule_df.to_csv(executed_schedule_csv, index=False)
    return executed_schedule_df


    
def generate_gantt_chart(source: Union[Path, str, pd.DataFrame], file_name: str, time_unit: str = 'm') -> None:
    """
    Generates a Gantt chart with two sections:
    - Upper: Last 5 experiments
    - Lower: All experiments
    Y-axis: Experiment name and UUID
    X-axis: Date and time with local timezone using astimezone
    """
    try:
        # 図と軸の設定
        fig, ax = plt.subplots(figsize=(10, 6))

        # スケジュールデータの読み込み
        df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

        # scheduled_timeをdatetimeに変換（分単位）
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'], unit=time_unit)
        df["processing_time"] = pd.to_timedelta(df["processing_time"], unit=time_unit)

        # ローカルタイムゾーンを取得
        local_timezone = datetime.now().astimezone().tzinfo
        print(f"Local timezone: {local_timezone}")

        time_diff = local_timezone.utcoffset(datetime.now())

        # scheduled_timeをローカルタイムゾーンに変換
        df['scheduled_time'] = df['scheduled_time'].apply(lambda x: x + time_diff)

        # scheduled_timeでソート
        df = df.sort_values('scheduled_time')

        # Y軸用のラベル作成
        df['colour_label'] = df['experiment_name'] + ": " + df['experiment_uuid']

        # processing_timeが分単位と仮定してend_timeを計算
        df['end_time'] = df['scheduled_time'] + pd.to_timedelta(df['processing_time'], unit='m')
        print(df['scheduled_time'])
        print(df['processing_time'])
        print(df['end_time'])

        # Gantt chartの描画
        # Chenge the colour depending on the colour_label
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        color_dict = {}
        for i, colour_label in enumerate(df['colour_label'].unique()):
            color_dict[colour_label] = colors[i % len(colors)]
        df['color'] = df['colour_label'].map(color_dict)

        for i, row in df.iterrows():
            ax.barh(0, row['processing_time'], left=row['scheduled_time'], color=row['color'])
        
        # X軸の設定
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
        plt.xlabel('Time')

        ax.yaxis.set_visible(False)

        ax.legend(df['colour_label'].unique(), loc='upper left')


        # 図の保存
        plt.savefig(file_name)

    except Exception as e:
        print(f"Error generating Gantt chart: {e}")
            

if __name__ == "__main__":
    DIR = Path("iPSsimulation/results/2024-11-13_real_round3")
    dir = DIR
    executed_schedule_csv = dir / "executed_schedule.csv"
    executed_schedule_csv = get_executed_schedule(dir, executed_schedule_csv)

    # generate_gantt_chart(executed_schedule_csv, "gantt_chart.png")
