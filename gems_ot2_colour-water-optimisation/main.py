from datetime import datetime
from pathlib import Path
import os
import inspect
import tempfile
import polars as pl

from gems_python.multi_machine_problem_interval_task.interactive_ui import PluginManager
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments



def test_initialise(self):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print(current_dir)
    with tempfile.TemporaryDirectory(dir=current_dir) as dir:
        current_unixtime = datetime.now()
        print(f"current_unixtime: {current_unixtime.timestamp()} ({current_unixtime.astimezone().isoformat()})")
        current_unixtime = int(current_unixtime.timestamp())

        experiments = Experiments(parent_dir_path=Path(dir), reference_time = current_unixtime)
        plugin_manager = PluginManager(experiments)

        # Copy ./minimum.py to dir/experimental_setting/minimum.py
        import shutil
        shutil.copy(f"{current_dir}/minimum.py", f"{dir}/experimental_setting/minimum.py")




        machines = \
r"""
0,Pippeting machine 1
0,Pippeting machine 2
1,Heating machine 1
2,Centrifuge machine 1

"""
        with open(f"{dir}/mode/mode_add_machines.txt", "w") as f:
            f.write(machines)

        mode = "add_machines"
        with open(f"{dir}/mode/mode.txt", "w") as f:
            f.write(mode)


        ex = "minimum.gen_standard_experiment"
        with open(f"{dir}/mode/mode_add_experiments.txt", "w") as f:
            f.write(ex)

        with open(f"{dir}/mode/mode_add_experiments.txt", "w") as f:
            f.write(ex)

        # mode = "add_experiments"
        # with open(f"{dir}/mode/mode.txt", "w") as f:
        #     f.write(mode)


        # プラグインマネージャーの開始前にリロードの選択を促す
        reload_choice = 'y' #input("実験をリロードしますか？ (y/n): ").strip().lower()
        if reload_choice == 'y':
            try:
                step = '' #input("リロードするステップを入力してください。空白の場合は自動的に最大ステップまでリロードします。: ").strip()
                if step == '':
                    step = None
                else:
                    step = int(step)
                experiments = experiments.reload(step)
            except ValueError:
                print("無効なステップ番号です。リロードをスキップします。")
            except Exception as err:
                print(f"リロード中にエラーが発生しました: {err}. リロードをスキップします。")

        plugin_manager = PluginManager(experiments)
        plugin_manager.run()




def main():
    dir = input("Enter the directory path for experiments: ").strip()
    if dir == '':
        dir = "ot2_experiment"
    print(dir)
    experiments = Experiments(parent_dir_path=Path(dir), reference_time = datetime.now().timestamp()//60)
    plugin_manager = PluginManager(experiments)

    # プラグインマネージャーの開始前にリロードの選択を促す
    reload_choice = 'y' #input("実験をリロードしますか？ (y/n): ").strip().lower()
    if reload_choice == 'y':
        try:
            step = '' #input("リロードするステップを入力してください。空白の場合は自動的に最大ステップまでリロードします。: ").strip()
            if step == '':
                step = None
            else:
                step = int(step)
            experiments = experiments.reload(step)
        except ValueError:
            print("無効なステップ番号です。リロードをスキップします。")
        except Exception as err:
            print(f"リロード中にエラーが発生しました: {err}. リロードをスキップします。")

    
    def set_task_group_ids(experiments):
        """
        Assign the task group ids.
        If the task group ids are not assigned, assign them.
        """
        used_ids = set()
        new_id = 0
        mp = dict()
        # experiment_uuid:id

        for task_group_index in range(len(experiments.task_groups)):
            experiments.task_groups[task_group_index].task_group_id = None

        for task_group in experiments.task_groups:
            if task_group.task_group_id is not None:
                used_ids.add(task_group.task_group_id)

        for task_group_index in range(len(experiments.task_groups)):
            if experiments.task_groups[task_group_index].task_group_id is None:
                while new_id in used_ids:
                    new_id += 1
                experiments.task_groups[task_group_index].task_group_id = new_id
                used_ids.add(new_id)
                mp[experiments.task_groups[task_group_index].experiment_uuid] = new_id

        for experiment_index in range(len(experiments.experiments)):
            experiments.experiments[experiment_index].current_task_group.task_group_id = mp[experiments.experiments[experiment_index].experiment_uuid]

        return experiments


    experiments = set_task_group_ids(experiments)

    # Reschedule?
    reschedule_choice = input("Reschedule experiments? (y/n): ").strip().lower()
    if reschedule_choice == 'y':
        reference_time = int(input("Enter the reference time: ").strip())
        scheduling_method = input("Enter the scheduling method: ").strip()
        experiments.set_task_group_ids()
        experiments.execute_scheduling(scheduling_method = scheduling_method, reference_time = reference_time)
    experiments.proceed_to_next_step()


    plugin_manager = PluginManager(experiments)
    plugin_manager.run()

if __name__ == "__main__":
    main()