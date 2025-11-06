from datetime import datetime, timedelta, timezone
import os
import json
from pathlib import Path
import pandas as pd

# cd the directory to this script is in
os.chdir(Path(__file__).parent)

# 結合するフォルダのパス
folder_path = "../../../GEMS-python-cell-culture/iPSsimulation/results/2024-11-13_real_round3"  # ここを適切なパスに変更してください
folder_path = Path(folder_path)

artifact_path = Path("../../data/241106_CCDS_253g1-hek293a_report/processed_data")
if not artifact_path.exists():
    artifact_path.mkdir(parents=True, exist_ok=True)
executed_task_path = artifact_path / "executed_task.csv"

if not os.path.exists(executed_task_path):
    # 各ステップフォルダの名前を取得
    step_folders = [folder_path / f for f in os.listdir(folder_path) if f.startswith("step_")]
    step_folders.sort()
    print(f"{step_folders=}")

    executed_task = pd.DataFrame()

    # データフレームのリスト

    for step_folder in step_folders:
        schedule_path = step_folder / "schedule.csv"
        experiment_result_path = step_folder / "experiment_result.json"
        try:
            schedule_df = pd.read_csv(schedule_path)
            with open(experiment_result_path) as f:
                experiment_result = json.load(f)

            task_group_id = experiment_result["task_group_id"]
            task_id = experiment_result["task_id"]
            task = schedule_df[(schedule_df["task_group_id"] == task_group_id) & (schedule_df["task_id"] == task_id)]
            

            print(f"{task=}")

            if len(task) > 1:
                print(f"Error: {task}")
                os.sleep(1)
                continue

            task = task.assign(**{key: value for key, value in experiment_result.items() if key not in task.columns})

            executed_task = pd.concat([executed_task, task])

        except FileNotFoundError:     
            print(f"File not found: {schedule_path}")
            continue
        
        except Exception as e:
            print(f"Error: {e}")
            continue

    executed_task.to_csv(executed_task_path, index=False)

executed_task = pd.read_csv(executed_task_path)

time_converted_task_path = artifact_path / "time_converted_task.csv"

if not time_converted_task_path.exists():
    # 作業時間の計算
    executed_task["scheduled_time"] = pd.to_datetime(executed_task["scheduled_time"], unit="m")
    executed_task["optimal_time_reference_time"] = pd.to_datetime(executed_task["optimal_time_reference_time"], unit="m")
    executed_task["processing_time_min"] = executed_task["processing_time"]

    executed_task.to_csv(time_converted_task_path, index=False)



# # データフレームのリスト

# for step_folder in step_folders:
#     pkl_path = os.path.join(folder_path, step_folder, "laboratory.pkl")
#     if os.path.exists(pkl_path):
#         print(f"{step_folder=}")
#         laboratory = Experiments.from_pickle(path = pkl_path)
#         # print(f"{laboratory=}")
#         earliest_task, eariest_group_id = TaskGroup.get_ealiest_task_in_task_groups(laboratory.task_groups)
#         # print(f"{earliest_task.to_dict()=}")
#         dict = earliest_task.to_dict()
#         dict["scheduled_time"] = int(dict["scheduled_time"])
#         dict["scheduled_time"] = dict["scheduled_time"] * 60
#         dict["scheduled_time"] = datetime.fromtimestamp(dict["scheduled_time"]).isoformat()
#         df = pl.DataFrame(
#             dict
#         )

#         #  processing_time ->  processing_time_min
#         df = df.rename({"processing_time": "processing_time_min"})
        
        
#         # print(f"{df=}")
#         task_schedule = task_schedule.vstack(df)


# task_schedule.write_csv(os.path.join(folder_path, "task_schedule.csv"))

# # Operation count
# operation_count = task_schedule.group_by("experiment_operation").count()
# operation_count.write_csv(os.path.join(folder_path, "operation_count.csv"))


# # def main():
# #     experiments = Experiments()
# #     experiments.run_experiment()

# # if __name__ == "__main__":
# #     main()