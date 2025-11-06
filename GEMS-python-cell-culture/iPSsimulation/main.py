from pathlib import Path
from gems_python.one_machine_problem_interval_task.interactive_ui import PluginManager
from gems_python.one_machine_problem_interval_task.transition_manager import Experiments

# 1731423600 2024_11_13_00_00_00 UNIX

UNIX_2024_11_13_00_00_00_IN_JP = 1731423600


def main():
    dir = input("Enter the directory path for experiments: ").strip()
    if dir == '':
        dir = "volatile"
    print(dir)
    experiments = Experiments(parent_dir_path=Path(dir), reference_time = (UNIX_2024_11_13_00_00_00_IN_JP//60) - 24*60*4)
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