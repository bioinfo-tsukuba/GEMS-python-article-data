import json
from datetime import datetime, timedelta
from pathlib import Path

def get_unix_time_with_unit(unit):
    now = datetime.now().timestamp()
    if unit == 's':
        return int(now)
    elif unit == 'm':
        return int(now//60)
    elif unit == 'h':
        return int(now//(60*60))
    elif unit == 'D':
        return int(now//(60*60*24))
    elif unit == 'M':
        return int(now//(60*60*24*30))
    elif unit == 'Y':
        return int(now//(60*60*24*365))
    else:
        raise ValueError("無効な単位です。s, m, h, D, M, Y のいずれかを指定してください。")
    

def create_experiment_result_json(
    task_response: str,
    task_group_id: int,
    task_id: int,
    optimal_time_reference_time: int = None,
    unit: str = None,
    result_path: str = None,
    save_json: bool = False,
    save_directory: str = None
) -> str:
    """
    タスク情報に基づいてJSONデータを生成し、必要に応じてファイルに保存する関数です。

    Args:
        task_response (str): "In Progress", "Completed", "Error" のいずれか。
        task_group_id (int): タスクグループの識別子。
        task_id (int): タスクの識別子。
        optimal_time_reference_time (int, optional):
            手動で指定するUnixタイム。Noneの場合、unitに基づき自動生成。
        unit (str, optional):
            optimal_time_reference_timeを自動生成する場合の単位（s, m, h, D, M, Y）。
        result_path (str, optional):
            task_responseが"Completed"の場合に必要な結果ファイルのパス。
        save_json (bool, optional):
            生成したJSONデータをファイルに保存するかどうか。
        save_directory (str, optional):
            JSONデータを保存するフォルダのパス。save_jsonがTrueの場合に必須。

    Returns:
        str: 生成されたJSON文字列。
    """
    # optimal_time_reference_timeが指定されていない場合、unitに基づいて自動生成
    if optimal_time_reference_time is None:
        if unit is not None:
            optimal_time_reference_time = get_unix_time_with_unit(unit)
            print(f"生成されたoptimal_time_reference_time: {optimal_time_reference_time}")
        else:
            raise ValueError("optimal_time_reference_timeを指定するか、unitを指定して自動生成してください。")

    # タスク応答に応じてデータ辞書を作成
    data = {
        "task_response": task_response,
        "task_group_id": task_group_id,
        "task_id": task_id,
        "optimal_time_reference_time": optimal_time_reference_time,
    }
    if task_response == "Completed":
        if result_path is None:
            raise ValueError("task_responseが'Completed'の場合、result_pathを指定してください。")
        data["result_path"] = result_path

    # JSONデータに変換
    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    print("生成されたJSONデータ:")
    print(json_data)

    # ファイルに保存する場合
    if save_json:
        if save_directory is None:
            raise ValueError("JSONを保存する場合、save_directoryを指定してください。")
        save_dir = Path(save_directory)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        file_path = save_dir / "experiment_result.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        print(f"JSONデータを {file_path} に保存しました。")

    return json_data

def main():
    task_response_options = ["In Progress", "Completed", "Error"]
    print("task_responseを選択してください:")
    for i, option in enumerate(task_response_options, 1):
        print(f"{i}: {option}")
    task_response_index = int(input("番号を入力: ")) - 1
    task_response = task_response_options[task_response_index]
    task_group_id = int(input("task_group_idを入力してください: "))
    task_id = int(input("task_idを入力してください: "))
    optimal_time_choice = input("optimal_time_reference_timeを自動生成しますか？ (y/n): ").lower()
    if optimal_time_choice == 'y':
        unit = input("単位を選択してください (s:秒, m:分, h:時間, D:日, M:月, Y:年): ").lower()
        optimal_time_reference_time = get_unix_time_with_unit(unit)
        print(f"生成されたoptimal_time_reference_time: {optimal_time_reference_time}")
    else:
        optimal_time_reference_time = int(input("optimal_time_reference_timeを手動で入力してください (Unix時間): "))

    if task_response == "In Progress":
        data = {
            "task_response": task_response,
            "task_group_id": task_group_id,
            "task_id": task_id,
            "optimal_time_reference_time": optimal_time_reference_time,
        }
    elif task_response == "Completed":
        result_path = input("result_pathを入力してください: ")
        data = {
            "task_response": task_response,
            "task_group_id": task_group_id,
            "task_id": task_id,
            "optimal_time_reference_time": optimal_time_reference_time,
            "result_path": result_path
        }

    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    print("生成されたJSONデータ:")
    print(json_data)

    save_choice = input("JSONデータを保存しますか？ (y/n): ").lower()
    if save_choice == 'y':
        save_path = Path(input("保存先のファイルパス（フォルダ）を入力してください: "))
        if not save_path.exists():
            save_path.mkdir(parents=True)
        save_path = save_path / "experiment_result.json"
        with open(save_path, 'w') as f:
            f.write(json_data)
        print(f"{save_path} に保存しました。")

if __name__ == "__main__":
    main()
