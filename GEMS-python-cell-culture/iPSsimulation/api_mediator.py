import asyncio
from pathlib import Path
import time
from uuid import UUID
from maholocon.client import Client
from maholocon.executor.protocol import Protocol
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

from cell_density_calculator import calc_simple_from_path
from samples.settings import API_PORT, IPS_EXPERIMENT_NAME, HEK_EXPERIMENT_NAME

from typing import List, Optional
from pydantic import BaseModel
import json
from datetime import datetime
import pickle
import argparse  # 追加

EXPERIMENT_DIR = Path('iPSsimulation/results/2024-11-13_real_round3')


HEK_PROCESSING_TIME = {
    "HEKExpire": 0,
    "HEKPassage": 120,
    "HEKGetImage": 10,
    "HEKSampling": 120,
    "HEKWaiting": 0,
    "LabwareRefill": 120,
}


IPS_PROCESSING_TIME = {
    "iPSExpire": 0,
    "iPSPassage": 120,
    "iPSGetImage": 10,
    "iPSMediumChange": 20,
    "iPSPlateCoating": 20,
    "iPSSampling": 120,
    "iPSWaiting": 0
}

class HEKExperimentManager(BaseModel):
    experiment_id: str
    cell_plate_id: str

    @classmethod
    def find_index_by_experiment_id(cls, hek_experiments: List['HEKExperimentManager'], experiment_id: str) -> Optional[int]:
        for i, hek_experiment in enumerate(hek_experiments):
            if hek_experiment.experiment_id == experiment_id:
                return i
        return None

    @classmethod
    def pre_passage_action(cls, hek_experiments: List['HEKExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(hek_experiments, experiment_id)
        if index is not None:
            return hek_experiments[index].cell_plate_id
        return None

    @classmethod
    def post_passage_action(cls, hek_experiments: List['HEKExperimentManager'], experiment_id: str, new_cell_plate_id: str) -> List['HEKExperimentManager']:
        index = cls.find_index_by_experiment_id(hek_experiments, experiment_id)
        if index is not None:
            hek_experiments[index].cell_plate_id = str(new_cell_plate_id)
        return hek_experiments

    @classmethod
    def pre_getimage_action(cls, hek_experiments: List['HEKExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(hek_experiments, experiment_id)
        if index is not None:
            return hek_experiments[index].cell_plate_id
        return None

    @classmethod
    def pre_sampling_action(cls, hek_experiments: List['HEKExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(hek_experiments, experiment_id)
        if index is not None:
            return hek_experiments[index].cell_plate_id
        return None

    # JSONへのダンプ（シリアライズ）
    @classmethod
    def dump_list(cls, hek_experiments: List['HEKExperimentManager'], file_path: str):
        """
        [
            {
                "experiment_id": "exp1",
                "cell_plate_id": "plateA"
            },
            {
                "experiment_id": "exp2",
                "cell_plate_id": "plateB"
            }
        ]
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([exp.model_dump() for exp in hek_experiments], f, ensure_ascii=False, indent=4)

    # JSONからのロード（デシリアライズ）
    @classmethod
    def load_list(cls, file_path: str) -> List['HEKExperimentManager']:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [cls(**item) for item in data]
    



class iPSExperimentManager(BaseModel):
    experiment_id: str
    cell_plate_id: str
    coated_plate_id: Optional[str] = None

    @classmethod
    def find_index_by_experiment_id(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> Optional[int]:
        for i, ips_experiment in enumerate(ips_experiments):
            if ips_experiment.experiment_id == experiment_id:
                return i
        return None

    @classmethod
    def pre_passage_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str):
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            if ips_experiments[index].coated_plate_id is None:
                print("coated_plate_id is None")
                return None
            else:
                return ips_experiments[index].cell_plate_id, ips_experiments[index].coated_plate_id
        return None

    @classmethod
    def post_passage_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> List['iPSExperimentManager']:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            ips_experiments[index].cell_plate_id = str(ips_experiments[index].coated_plate_id)
            ips_experiments[index].coated_plate_id = None
        return ips_experiments

    @classmethod
    def pre_getimage_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            return ips_experiments[index].cell_plate_id
        return None

    @classmethod
    def pre_medium_change_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            return ips_experiments[index].cell_plate_id
        return None
    
    @classmethod
    def pre_plate_coating_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            return ips_experiments[index].cell_plate_id
        return None

    @classmethod
    def post_plate_coating_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str, coated_plate_id: str) -> List['iPSExperimentManager']:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            ips_experiments[index].coated_plate_id = str(coated_plate_id)
        return ips_experiments

    @classmethod
    def pre_sampling_action(cls, ips_experiments: List['iPSExperimentManager'], experiment_id: str) -> Optional[str]:
        index = cls.find_index_by_experiment_id(ips_experiments, experiment_id)
        if index is not None:
            return ips_experiments[index].cell_plate_id
        return None

    # JSONへのダンプ（シリアライズ）
    @classmethod
    def dump_list(cls, ips_experiments: List['iPSExperimentManager'], file_path: str):
        """
        [
            {
                "experiment_id": "exp1",
                "cell_plate_id": "plateX",
                "coated_plate_id": "coat1"
            },
            {
                "experiment_id": "exp2",
                "cell_plate_id": "plateY",
                "coated_plate_id": null
            }
        ]
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([exp.model_dump() for exp in ips_experiments], f, ensure_ascii=False, indent=4)

    # JSONからのロード（デシリアライズ）
    @classmethod
    def load_list(cls, file_path: str) -> List['iPSExperimentManager']:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [cls(**item) for item in data]
    
class WholeManager:
    path: Path
    step: int = 0
    client = Client('http://localhost:' + str(API_PORT) + '/api/v1')
    ips_experiments: List[iPSExperimentManager] = None
    hek_experiments: List[HEKExperimentManager] = None

    def __init__(self, path: Path, ips_experiments: List[iPSExperimentManager] = None, hek_experiments: List[HEKExperimentManager] = None):
        self.path = path
        self.ips_experiments = ips_experiments
        self.hek_experiments = hek_experiments
        if ips_experiments is None and hek_experiments is None:
            self.refresh_experiment_data()

    def refresh_experiment_data(self):
        try:
            self.ips_experiments = iPSExperimentManager.load_list(self.get_current_step_dir() / "ips_experiments.json")
        except FileNotFoundError:
            self.ips_experiments = []
            print("ips_experiments.json not found")
        try:
            self.hek_experiments = HEKExperimentManager.load_list(self.get_current_step_dir() / "hek_experiments.json")
        except FileNotFoundError:
            self.hek_experiments = []
            print("hek_experiments.json not found")
        print(f"Refreshed Experiment Data from JSON {self.get_current_step_dir() / 'ips_experiments.json'} and {self.get_current_step_dir() / 'hek_experiments.json'}")
        print(f"ips_experiments: {self.ips_experiments}")
        print(f"hek_experiments: {self.hek_experiments}")

    def get_current_step_dir(self) -> Path:
        step_dir = self.path / f"step_{str(self.get_current_step()).zfill(8)}"
        return step_dir
    
    def get_minute_time(self) -> int:
        unix_time = int(datetime.now().timestamp())
        # Convert to minutes
        return unix_time // 60
        

    async def run_protocol(self, experiment: str, protocol: str, args: dict[str, UUID]):
        arg_str = ', '.join([f'{k}={v}' for k, v in args.items()])
        print(f'run {experiment}/{protocol}({arg_str})')
        async for res in self.client.run(experiment, protocol, args):
            if res.type == 'progress':
                print(res.data.progress)
            elif res.type == 'error':
                print('error')
                print(res.data.error)
            else:
                print(res)
        print()
        return res

    def run(self, experiment: str, protocol: str, args: dict[str, UUID]=None):
        if args is None:
            args = {}
        return asyncio.run(self.run_protocol(experiment, protocol, args))

    def write_result(self, res):
        print(res)

    def get_current_step(self) -> int:
        # 保存ディレクトリ内のすべてのstepディレクトリを取得
        step_dirs = [
            d for d in self.path.iterdir()
            if d.is_dir() and d.name.startswith('step_') and d.name[5:].isdigit()
        ]
        if not step_dirs:
            raise ValueError("リロード可能なステップディレクトリが見つかりません。")
        # ステップ番号を抽出し、最大値を選択
        step_numbers = [int(d.name[5:]) for d in step_dirs]
        step = max(step_numbers)
        current_step = int(step)
        return current_step
    
    def update_step(self, step = None):
        if step is None:
            current_step = self.get_current_step()
            self.step = current_step
        else:
            self.step = step
        

    def read_schedule(self):
        current_step = self.get_current_step()
        if current_step <= self.step:
            print("Already finished, or outputted")
            return None
        
        current_step_dir = self.get_current_step_dir()

        # Read schedule.csv
        schedule_csv = current_step_dir / "schedule.csv"
        df = pd.read_csv(schedule_csv)

        return df
    
    def get_top_schedule(self):
        df = self.read_schedule()
        if df is None:
            return None

        # sort by scheduled_time
        df = df.sort_values('scheduled_time')

        # Get top row
        top_row = df.iloc[0]

        return top_row
    

    def visualize_experiments(self, file_name: str = "plate_info"):
        # Convert the experiment data to DataFrames for easier plotting
        ips_data = pd.DataFrame([exp.model_dump() for exp in self.ips_experiments])
        hek_data = pd.DataFrame([exp.model_dump() for exp in self.hek_experiments])

        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot iPSExperimentManager data
        if not ips_data.empty:
            axes[0].table(
                cellText=ips_data.values,
                colLabels=ips_data.columns,
                cellLoc='center',
                loc='center'
            )
            axes[0].axis('off')
            axes[0].set_title('iPS Experiments', pad=20)
        else:
            axes[0].text(0.5, 0.5, 'No iPS Experiments', ha='center', va='center', fontsize=12)
            axes[0].axis('off')

        # Plot HEKExperimentManager data
        if not hek_data.empty:
            axes[1].table(
                cellText=hek_data.values,
                colLabels=hek_data.columns,
                cellLoc='center',
                loc='center'
            )
            axes[1].axis('off')
            axes[1].set_title('HEK Experiments', pad=20)
        else:
            axes[1].text(0.5, 0.5, 'No HEK Experiments', ha='center', va='center', fontsize=12)
            axes[1].axis('off')

        plt.tight_layout()
        save_dir = self.path / "step_current"
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリの存在を確認・作成
            fig.savefig(save_dir / f"{file_name}.png", bbox_inches='tight')
            fig.savefig(save_dir / f"{file_name}.pdf", bbox_inches='tight')
            fig.savefig(save_dir / f"{file_name}.svg", bbox_inches='tight')
            save_dir = self.get_current_step_dir()
            fig.savefig(save_dir / f"{file_name}.png", bbox_inches='tight')
            fig.savefig(save_dir / f"{file_name}.pdf", bbox_inches='tight')
            fig.savefig(save_dir / f"{file_name}.svg", bbox_inches='tight')
            plt.close(fig)  # メモリ解放
        else:
            plt.show()
    
    def generate_gantt_chart(self, file_name: str = "schedule_vis"):
        """
        Generates a Gantt chart with two sections:
        - Upper: Last 5 experiments
        - Lower: All experiments
        Y-axis: Experiment name and UUID
        X-axis: Date and time with local timezone using astimezone
        """
        try:
            # 図と軸の設定
            fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
                3, 1, figsize=(18, 12),
                gridspec_kw={'height_ratios': [1, 2, 5]}
            )

            # スケジュールデータの読み込み
            df = self.read_schedule()

            # scheduled_timeをdatetimeに変換（分単位）
            df['scheduled_time'] = pd.to_datetime(df['scheduled_time'], unit='m', utc=True)

            # ローカルタイムゾーンを取得
            local_timezone = datetime.now().astimezone().tzinfo
            print(f"Local timezone: {local_timezone}")

            time_diff = local_timezone.utcoffset(datetime.now())

            # scheduled_timeをローカルタイムゾーンに変換
            df['scheduled_time'] = df['scheduled_time'].apply(lambda x: x + time_diff)

            # scheduled_timeでソート
            df = df.sort_values('scheduled_time')

            # Y軸用のラベル作成
            df['y_label'] = df['experiment_name'] + ": " + df['experiment_uuid']

            # processing_timeが分単位と仮定してend_timeを計算
            df['end_time'] = df['scheduled_time'] + pd.to_timedelta(df['processing_time'], unit='m')
            print(df['scheduled_time'])
            print(df['processing_time'])
            print(df['end_time'])
            

            # すべての実験データ
            all_experiments = df.copy()

            # 直近5つの実験データ
            next_five_experiments = df.iloc[:5]

            # 直近1つの実験データ
            next_experiment = df.iloc[0]

            # ガントバーをプロットする関数
            def plot_gantt(ax, data, color):
                for _, row in data.iterrows():
                    ax.barh(
                        row['y_label'],
                        width=row['processing_time']/(60*24),
                        left=row['scheduled_time'],
                        height=0.4,
                        color=color,
                        edgecolor='black'
                    )

            # 下部にすべての実験をプロット
            plot_gantt(ax_bottom, all_experiments, color='skyblue')
            ax_bottom.set_title('All Experiments')
            ax_bottom.set_xlabel('Date and Time')
            ax_bottom.set_ylabel('Experiment name: Experiment uuid')
            ax_bottom.invert_yaxis()  # 直近の実験が上に来るようにする
            ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.7)

            # 中部に直近5つの実験をプロット
            plot_gantt(ax_middle, next_five_experiments, color='salmon')
            ax_middle.set_title('Next 5 Experiments')
            ax_middle.set_ylabel('Experiment name: Experiment uuid')
            ax_middle.invert_yaxis()  # 下部と整合性を取る
            ax_middle.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # 中部のプロットには、それぞれの時間と処理名を書く
            for _, row in next_five_experiments.iterrows():        # 各バーにテキストを追加
                ax_middle.text(
                    row['scheduled_time'], row['y_label'],
                    f"{row['experiment_operation']} \n@ {row['scheduled_time'].strftime('%H:%M')}",
                    va='center', ha='left', fontsize=9
                )

            # 上部に直近1つの実験をプロット
            ax_top.barh(
                next_experiment['y_label'],
                width=next_experiment['processing_time']/(60*24),
                left=next_experiment['scheduled_time'],
                height=0.4,
                color='lightgreen',
                edgecolor='black'
            )
            ax_top.set_title('Next Experiment')
            ax_top.set_ylabel('Experiment name: Experiment uuid')
            ax_top.invert_yaxis()  # 下部と整合性を取る
            ax_top.grid(True, axis='x', linestyle='--', alpha=0.7)

            ax_top.text(
                next_experiment['scheduled_time'], next_experiment['y_label'],
                f"{next_experiment['experiment_operation']} \n@ {next_experiment['scheduled_time'].strftime('%H:%M')}",
                va='center', ha='left', fontsize=9
            )

            # X軸のフォーマット設定
            date_format = DateFormatter("%Y-%m-%d %H:%M")
            ax_top.xaxis.set_major_formatter(date_format)
            ax_middle.xaxis.set_major_formatter(date_format)
            ax_bottom.xaxis.set_major_formatter(date_format)
            plt.setp(ax_bottom.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            # 図の保存
            save_dir = self.path / "step_current"
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリの存在を確認・作成
                fig.savefig(save_dir / f"{file_name}.png", bbox_inches='tight')
                fig.savefig(save_dir / f"{file_name}.pdf", bbox_inches='tight')
                fig.savefig(save_dir / f"{file_name}.svg", bbox_inches='tight')
                all_experiments.to_csv(save_dir / "mediator_schedule.csv", index = False)
                save_dir = self.get_current_step_dir()
                fig.savefig(save_dir / f"{file_name}.png", bbox_inches='tight')
                fig.savefig(save_dir / f"{file_name}.pdf", bbox_inches='tight')
                fig.savefig(save_dir / f"{file_name}.svg", bbox_inches='tight')
                all_experiments.to_csv(save_dir / "mediator_schedule.csv", index = False)
                plt.close(fig)  # メモリ解放
            else:
                plt.show()

        except Exception as e:
            print(f"Error generating Gantt chart: {e}")
    
    def protocol_processor(self, schedule) -> pd.DataFrame:
        current_time = datetime.now().astimezone()
        scheduled_time = datetime.fromtimestamp(schedule['scheduled_time']*60).astimezone()
        experiment_uuid = schedule['experiment_uuid']
        experiment_operation = schedule['experiment_operation']
        
        if experiment_operation in HEK_PROCESSING_TIME:
            df = self.hek_protocol_processor(schedule)
        elif experiment_operation in IPS_PROCESSING_TIME:
            df = self.ips_protocol_processor(schedule)
        else:
            pass

        return df

    def hek_protocol_processor(self, schedule) -> pd.DataFrame:
        # EXPERIMENT_NAMEが必要になる
        df = pd.DataFrame()
        df['operation'] = [schedule['experiment_operation']]
        df['time'] = [self.get_minute_time()]

        match schedule['experiment_operation']:
            case "HEKExpire":
                pass
            case "HEKPassage":
                # cell_plate_idを取得
                cell_plate = HEKExperimentManager.pre_passage_action(self.hek_experiments, schedule['experiment_uuid'])
                res = self.run(HEK_EXPERIMENT_NAME, 'passage_hek293a', {'cell_plate': cell_plate})
                new_plate_id = res.data.allocateds['empty_plate']
                self.hek_experiments = HEKExperimentManager.post_passage_action(self.hek_experiments, schedule['experiment_uuid'], new_plate_id)
            case "HEKGetImage":
                # cell_plate_idを取得
                cell_plate = HEKExperimentManager.pre_getimage_action(self.hek_experiments, schedule['experiment_uuid'])
                res = self.run(HEK_EXPERIMENT_NAME, 'getimage', {'cell_plate': cell_plate})
                data_path = res.data.observeds[0]
                extension = data_path.split('.')[-1]
                save_path = self.get_current_step_dir() / ("get_image_figure." + extension)
                print(f"Saving image to {save_path}")
                with open(save_path, 'wb') as f:
                    self.client.get_artifact(data_path, f)
                # TODO: calc_density
                file_name = save_path.name
                dir = save_path.parent
                try:
                    density = calc_simple_from_path(str(save_path), str(dir))
                    df['density'] = [float(density)]
                except Exception as e:
                    print(f"Error calculating density: {e}")
                    density = None
                    df['note'] = ["Error_calculating_density"]
                """
                cell_plate = root.get(cell_plate.id)
                res = run(exp_name, 'getimage', {'cell_plate': cell_plate.id})
                data_path = res.data.observeds[0]
                extension = data_path.split('.')[-1]
                """

            case "HEKSampling":
                """
                res = run(exp_name, 'sampling', {'cell_plate': empty_plate_id})
                print(res)
                """
                # cell_plate_idを取得
                cell_plate = HEKExperimentManager.pre_sampling_action(self.hek_experiments, schedule['experiment_uuid'])
                res = self.run(HEK_EXPERIMENT_NAME, 'sampling', {'cell_plate': cell_plate})
                pass

            case "HEKWaiting":
                pass

            case _:
                pass

        return df

    def ips_protocol_processor(self, schedule) -> pd.DataFrame:
        # EXPERIMENT_NAMEが必要になる
        df = pd.DataFrame()
        df['operation'] = [schedule['experiment_operation']]
        df['time'] = [self.get_minute_time()]
        match schedule['experiment_operation']:
            case "iPSExpire":
                pass
            case "iPSPassage":
                """
                # passage
                res = run(exp_name, 'passage_253g1', {'cell_plate': cell_plate.id, 'coated_plate': coated_plate_id})
                coated_plate_id = res.data.allocateds['coated_plate']
                """
                # cell_plate_idを取得
                cell_plate, coated_plate = iPSExperimentManager.pre_passage_action(self.ips_experiments, schedule['experiment_uuid'])
                res = self.run(IPS_EXPERIMENT_NAME, 'passage_253g1', {'cell_plate': cell_plate, 'coated_plate': coated_plate})
                coated_plate_id = res.data.allocateds['coated_plate']
                self.ips_experiments = iPSExperimentManager.post_passage_action(self.ips_experiments, schedule['experiment_uuid'])

                pass
            case "iPSGetImage":
                """

                # getimage
                root = client.get_tree()
                cell_plate = root.get(cell_plate.id)
                res = run(exp_name, 'getimage', {'cell_plate': cell_plate.id})
                data_path = res.data.observeds[0]
                extension = data_path.split('.')[-1]
                with open('data.' + extension, 'wb') as f:
                    client.get_artifact(data_path, f)
                """
                # cell_plate_idを取得
                cell_plate = iPSExperimentManager.pre_getimage_action(self.ips_experiments, schedule['experiment_uuid'])
                res = self.run(IPS_EXPERIMENT_NAME, 'getimage', {'cell_plate': cell_plate})
                data_path = res.data.observeds[0]
                extension = data_path.split('.')[-1]
                save_path = self.get_current_step_dir() / ("get_image_figure." + extension)
                with open(save_path, 'wb') as f:
                    self.client.get_artifact(data_path, f)
                # TODO: calc_density
                file_name = save_path.name
                dir = save_path.parent
                try:
                    density = calc_simple_from_path(str(save_path), str(dir))
                    df['density'] = [float(density)]
                except Exception as e:
                    print(f"Error calculating density: {e}")
                    density = None
                    df['note'] = ["Error_calculating_density"]

            case "iPSMediumChange":
                # res = run(exp_name, 'mediumchange', {'cell_plate': cell_plate.id})
                # cell_plate_idを取得
                cell_plate = iPSExperimentManager.pre_medium_change_action(self.ips_experiments, schedule['experiment_uuid'])
                res = self.run(IPS_EXPERIMENT_NAME, 'mediumchange', {'cell_plate': cell_plate})

            case "iPSPlateCoating":
                """
                res = run(exp_name, 'platecoating')
                coated_plate_id = res.data.allocateds['coated_plate']
                print(coated_plate_id)
                coated_plate = client.get_tree(coated_plate_id)
                print(coated_plate)
                """
                # cell_plate_idを取得
                cell_plate = iPSExperimentManager.pre_plate_coating_action(self.ips_experiments, schedule['experiment_uuid'])
                res = self.run(IPS_EXPERIMENT_NAME, 'platecoating')
                coated_plate_id = res.data.allocateds['coated_plate']
                self.ips_experiments = iPSExperimentManager.post_plate_coating_action(self.ips_experiments, schedule['experiment_uuid'], coated_plate_id)

            case "iPSSampling":
                """
                # sampling
                res = run(exp_name, 'sampling', {'cell_plate': coated_plate_id})
                print(res)
                """
                # cell_plate_idを取得
                cell_plate = iPSExperimentManager.pre_sampling_action(self.ips_experiments, schedule['experiment_uuid'])
                res = self.run(IPS_EXPERIMENT_NAME, 'sampling', {'cell_plate': cell_plate})

            case "iPSWaiting":
                pass

            case _:
                pass

        return df

    def make_output(self, task_group_id: int, task_id: int, optimal_time_reference_time: int = None, result_df:pd.DataFrame = None, task_response:str = "success", result_path: str = None):
        """
        Example of
        experiment_result.json:
        if task is successful:
        {
            "task_response": "success",
            "task_group_id": 0,
            "task_id": 0,
            "optimal_time_reference_time": 0,
            "result_path": "result.csv"
        }

        minimum required fields:
        {
            "task_group_id": 0,
            "task_id": 0,
            "optimal_time_reference_time": 0,
            "result_path": "result.csv"
        }

        else:
        {
            "task_response": "error",
            "task_group_id": 0,
            "task_id": 0,
            "optimal_time_reference_time": 0
        }
        """

        if optimal_time_reference_time is None:
            optimal_time_reference_time = self.get_minute_time()

        match task_response:
            case "success":
                if result_df is None:
                    result_df = pd.DataFrame()
                if result_path is None:
                    result_path = self.get_current_step_dir() / "result.csv"
                    result_df.to_csv(result_path, index=False)
                output = {
                    "task_response": task_response,
                    "task_group_id": int(task_group_id),
                    "task_id": int(task_id),
                    "optimal_time_reference_time": int(optimal_time_reference_time),
                    "result_path": str(result_path)
                }
            case "error":
                output = {
                    "task_response": task_response,
                    "task_group_id": int(task_group_id),
                    "task_id": int(task_id),
                    "optimal_time_reference_time": int(optimal_time_reference_time)
                }
            case _:
                output = {
                    "task_response": "error",
                    "task_group_id": int(task_group_id),
                    "task_id": int(task_id),
                    "optimal_time_reference_time": int(optimal_time_reference_time)
                }
            
        # Save the iPSExperimentManager and HEKExperimentManager
        iPSExperimentManager.dump_list(self.ips_experiments, self.get_current_step_dir() / "ips_experiments.json")
        HEKExperimentManager.dump_list(self.hek_experiments, self.get_current_step_dir() / "hek_experiments.json")

        # Save as pkls
        with open(self.get_current_step_dir() / "api_mediator.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        # Write to experiment_result_temp.json
        with open(self.get_current_step_dir() / "experiment_result_temp.json", 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        # Rename experiment_result_temp.json to experiment_result.json
        experiment_result_temp = self.get_current_step_dir() / "experiment_result_temp.json"
        experiment_result = self.get_current_step_dir() / "experiment_result.json"
        experiment_result_temp.rename(experiment_result)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the iPS simulation mediator.")
    parser.add_argument('-m', '--manual', action='store_true', help="Enable manual mode to control protocol execution.")
    args = parser.parse_args()
    print(args)

    # JSONからロード
    loaded_ips_list = iPSExperimentManager.load_list(input("iPS Experiment JSON file path: "))
    iPSExperimentManager.dump_list(loaded_ips_list, 'ips_experiments.json')
    print(f"{loaded_ips_list=}")

    # JSONからロード
    loaded_hek_list = HEKExperimentManager.load_list(input("HEK Experiment JSON file path: "))
    HEKExperimentManager.dump_list(loaded_hek_list, 'hek_experiments.json')
    print(loaded_hek_list)

    input("Press Enter to start api mediator")

    # WholeManagerの使用例
    whole_manager = WholeManager(Path(EXPERIMENT_DIR), ips_experiments=loaded_ips_list, hek_experiments=loaded_hek_list)

    while True:
        # Read schedule
        schedule = whole_manager.get_top_schedule()
        if schedule is None:
            print("No Schedule")

        else:
            print("Schedule", schedule, sep='\n')

            # Get Current Time
            current_time = datetime.now().astimezone()
            whole_manager.generate_gantt_chart()
            whole_manager.visualize_experiments()
            # Get Scheduled Time
            # The scheduled_time is in minutes of UNIX time
            # Convert it to datetime
            scheduled_time = datetime.fromtimestamp(schedule['scheduled_time']*60).astimezone()
            experiment_uuid = schedule['experiment_uuid']
            experiment_operation = schedule['experiment_operation']
            task_group_id = schedule['task_group_id']
            task_id = schedule['task_id']
            print(f"{current_time=}")
            print(f"{scheduled_time=}")


            # 変更後の条件分岐
            if args.manual:
                user_input = input("Run the next protocol? Yes or No [Y/N]: ")
                print(f"{user_input=}")
                if user_input in ['Y', 'y', 'Yes', 'yes']:
                    whole_manager.update_step()
                    # Run protocol
                    df = whole_manager.protocol_processor(schedule)
                    print(df)
                    # Make output
                    output = whole_manager.make_output(task_group_id, task_id, result_df=df)

                    print("Run Protocol")
                else:
                    print("Skipped protocol execution based on user input.")
            else:
                # Compare current time and scheduled time
                if current_time >= scheduled_time:
                    whole_manager.update_step()
                    # Run protocol
                    df = whole_manager.protocol_processor(schedule)
                    print(df)
                    # Make output
                    output = whole_manager.make_output(task_group_id, task_id, result_df=df)


                    print("Run Protocol")
                    # break

                else:
                    print("Wait")

        # Wait for 1 minute
        time.sleep(10)
