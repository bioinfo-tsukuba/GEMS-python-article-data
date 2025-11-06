from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import polars as pl

from gen_result_json import create_experiment_result_json

def get_python_executable(venv_path):
    """
    仮想環境のPython実行ファイルのパスを取得します。
    """
    if os.name == 'nt':  # Windowsの場合
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Macの場合
        return os.path.join(venv_path, 'bin', 'python')
    
def run_main(folder, args=None):
    """
    指定されたフォルダ内のmain.pyを、そのフォルダで、そのフォルダの.venvを使用して実行します。
    
    :param folder: 実行するフォルダ名
    :param args: main.pyに渡す引数のリスト（オプション）
    """
    folder = Path(folder)

    # フォルダが存在するか確認
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return
    
    venv_path = str(folder/'.venv')
    python_exe = get_python_executable(venv_path)
    main_py = str(folder/'main.py')
    
    # 仮想環境のPython実行ファイルが存在するか確認
    if not os.path.exists(python_exe):
        print(f"Python executable not found in {venv_path}")
        return
    
    # main.pyが存在するか確認
    if not os.path.exists(main_py):
        print(f"main.py not found in {folder}")
        return
    
    # 実行コマンドを構築
    cmd = [python_exe, main_py]
    
    # 引数がある場合はコマンドに追加
    if args:
        cmd.extend(args)
    
    # subprocessで実行
    try:
        print(f"Running {main_py} with {python_exe} and arguments: {args}")
        result = subprocess.run(cmd, check=True)
        print(f"{main_py} finished with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {main_py}: {e}")

def run_opentrons(file_path: Path, simulate: bool = True):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    # 実行コマンドを構築
    if simulate:
        cmd = ['opentrons_simulate', str(file_path)]
    else:
        cmd = ['opentrons_execute', str(file_path)]
    
    # subprocessで実行
    try:
        print(f"Running OpenTrons protocol: {file_path}")
        result = subprocess.run(cmd, check=True)
        print(f"OpenTrons protocol finished with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running OpenTrons protocol: {e}")

def main(simulate=True):
    # OpenTronsプロトコルの実行
    file_path = Path('test/ot2_sim.py')
    run_opentrons(file_path, simulate=simulate)

    if simulate:
        print("Running simulations:")

        # 実行したいフォルダのリスト
        folders = ['color_water_simulator']
        
        for folder in folders:
            run_main(folder)

def get_latest_step_dir(experiment_dir_path: Path = Path('./ot2_experiment')):
    """
    指定されたフォルダ内の最新のステップフォルダのパスを取得します。
    
    :param experiment_dir_path: 実験フォルダのパス
    :return: 最新のステップフォルダのパス
    """
    # ステップフォルダのリストを取得(step_00000001, step_00000002, ...)
    # 正規表現でそれ以外のフォルダは無視
    step_dirs: list[Path] = [d for d in experiment_dir_path.iterdir() if d.is_dir() and re.match(r'step_\d{8}', d.name)]

    # ない場合はNoneを返す
    if not step_dirs:
        return None
    
    # ステップフォルダのリストをソート
    step_dirs.sort()

    # 最新のステップフォルダを返す
    return step_dirs[-1]

def one_step():
    step_dir: None | Path = get_latest_step_dir()
    if step_dir:
        schedule: Path = step_dir / 'schedule.csv'
        df: pl.DataFrame = pl.read_csv(schedule)
        # experiment_name	experiment_uuid	task_group_id	task_id	processing_time	experiment_operation	scheduled_time
        # Smallest scheduled_time
        next_operation = df.sort('scheduled_time').head(n=1)
        task_group_id = next_operation.get_column('task_group_id')[0]
        task_id = next_operation.get_column('task_id')[0]
        task_status = next_operation.get_column('task_status')[0]
        print(f"Next task_group_id: {task_group_id}, task_id: {task_id}")
        print(f"Next operation: {next_operation}")

        if task_status == "NOT_STARTED":
            print("Starting task...")
            # Start task
            # task_group_id, task_idを使って実行
            # 例: run_task(task_group_id, task_id)
        elif task_status == "IN_PROGRESS":
            print("Task in progress...")
            # Check task progress
            # task_group_id, task_idを使って進捗を確認
            # 例: check_task_progress(task_group_id, task_id)
    else:
        print("No step directories found.")
    # opentrons_main = "opentron_main"
    # run_main(opentrons_main)
    # main()

def generate_ot2_code(ratio_file_abs_path: str, start_column: int):
    ratio_file_abs_path = str(ratio_file_abs_path)
    start_column = str(start_column)
    folder = "./ot2_code_generator/"
    cmd = ["-t","./ot2_code_generator/ot_2_template.py", "-o","/Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/opentron_main/ot2_code.py", "-r", ratio_file_abs_path, "-s", start_column, "-c"]
    run_main(folder=folder, args=cmd)

def run_ot2_code():
    folder = "./opentron_main/"
    cmd = []
    run_main(folder=folder, args=cmd)

def run_RGB_calc(output_directory: str, start_column: int, ratio_file_abs_path: str):
    folder = "./RGB_converter/"
    cmd = ["-o", str(output_directory), "-s", str(start_column), "-r", str(ratio_file_abs_path)]
    run_main(folder=folder, args=cmd)

def get_latest_folder(folder: Path = Path('./ot2_experiment')):
    """
    指定されたフォルダ内の最新のフォルダのパスを取得します。
    
    :param folder: フォルダのパス
    :return: 最新のフォルダのパス
    step_00000001, step_00000002, ...
    """

    # フォルダのリストを取得
    # 正規表現でそれ以外のフォルダは無視
    dirs: list[Path] = [d for d in folder.iterdir() if d.is_dir() and re.match(r'step_\d{8}', d.name)]

    # ない場合はNoneを返す
    if not dirs:
        return None
    
    # フォルダのリストをソート
    dirs.sort()

    # 最新のフォルダを返す
    return dirs[-1]

def get_ratio_file_path(step_dir: Path):
    """
    指定されたステップフォルダ内のratioファイルのパスを取得します。
    
    :param step_dir: ステップフォルダのパス
    :return: ratioファイルのパス
    """
    # scheduleファイルのパスを取得
    schedule: Path = step_dir / 'schedule.csv'
    df: pl.DataFrame = pl.read_csv(schedule)

    # ratioファイルのパスを取得
    ratio_file_path = df.get_column("experiment_operation")[0]
    return ratio_file_path



if __name__ == '__main__':
    # one_step()
    print(get_latest_folder())
    start_column = 1
    STEP = 3
    for i in range(STEP):
        print(f"Step {i+1}/{STEP} starting...")
        # OT2:Start
        print(f"Starting OT2 code generation with start_column={start_column}")
        save_directory_ot2_start= Path(get_latest_step_dir())
        ratio_file_path = Path(get_ratio_file_path(save_directory_ot2_start))
        ratio_file_abs_path = ratio_file_path.absolute()
        print(f"{ratio_file_abs_path=}")
        print(f"{start_column=}")
        generate_ot2_code(ratio_file_abs_path, start_column)
        print(f"{save_directory_ot2_start=}")
        create_experiment_result_json(
            task_response="In Progress",
            task_group_id=0,
            task_id=0,
            optimal_time_reference_time=None,
            unit="m",
            result_path=None,
            save_json=True,
            save_directory=str(save_directory_ot2_start)
        )
        while save_directory_ot2_start == Path(get_latest_step_dir()):
            time.sleep(1)

        save_directory_ot2_end = Path(get_latest_step_dir())
        print(f"{save_directory_ot2_end=}")
        result_path = save_directory_ot2_end/"dummy_result.csv"
        df = pl.DataFrame({
            "time": [datetime.now().timestamp()//60],
        })
        run_ot2_code()
        df.write_csv(file=str(result_path))
        create_experiment_result_json(
            task_response="Completed",
            task_group_id=0,
            task_id=0,
            optimal_time_reference_time=None,
            unit="m",
            result_path=str(result_path),
            save_json=True,
            save_directory=str(save_directory_ot2_end)
        )
        while save_directory_ot2_end == Path(get_latest_step_dir()):
            time.sleep(1)
        print(f"OT2 code generation completed for step {i+1}/{STEP}")
        # OT2:End

        # RGB_calc:Start
        print(f"Starting RGB_calc with start_column={start_column}")
        save_directory_RGB_calc_start = Path(get_latest_step_dir())
        print(f"{save_directory_RGB_calc_start=}")
        create_experiment_result_json(
            task_response="In Progress",
            task_group_id=0,
            task_id=1,
            optimal_time_reference_time=None,
            unit="m",
            result_path=None,
            save_json=True,
            save_directory=str(save_directory_RGB_calc_start)
        )
        while save_directory_RGB_calc_start == Path(get_latest_step_dir()):
            time.sleep(1)

        save_directory_RGB_calc_end = Path(get_latest_step_dir())
        print(f"{save_directory_RGB_calc_end=}")
        result_path = save_directory_RGB_calc_end/"merged.csv"
        run_RGB_calc(str(save_directory_RGB_calc_end), start_column, str(ratio_file_abs_path))
        create_experiment_result_json(
            task_response="Completed",
            task_group_id=0,
            task_id=1,
            optimal_time_reference_time=None,
            unit="m",
            result_path=str(result_path),
            save_json=True,
            save_directory=str(save_directory_RGB_calc_end)
        )
        while save_directory_RGB_calc_end == Path(get_latest_step_dir()):
            time.sleep(1)
        print(f"RGB_calc completed for step {i+1}/{STEP}")
        # RGB_calc:End

        start_column += 4