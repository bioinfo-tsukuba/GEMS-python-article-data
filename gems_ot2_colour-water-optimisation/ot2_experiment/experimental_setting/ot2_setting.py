from datetime import datetime
from pathlib import Path
from uuid import uuid4
import polars as pl
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.multi_machine_problem_interval_task.task_info import TaskGroup, Task, Machine
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import NonePenalty, LinearPenalty

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement, qLogExpectedImprovement  # qLogExpectedImprovement を利用
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

UNIX_2038_01_01_00_00_00 = 2145916800
UNIX_2038_01_01_00_00_00_minutes = UNIX_2038_01_01_00_00_00 // 60
TARGET_DATA_PATH = Path("TARGRT_DATA/combined.csv")


"""
    processing_time: int  # タスクの処理時間
    experiment_operation: str
    optimal_machine_type: int # タスクの最適なマシンタイプ
    interval: int = field(default=0)        # タスク間のインターバル、最初のタスクにはインターバルはない
    task_status: TaskStatus = TaskStatus.NOT_STARTED  # タスクのステータス
    allocated_machine_id: int = field(default=None)  # タスクが割り当てられたマシンのID
    scheduled_time: int = field(default=None)  # タスクの開始時刻
    task_id: int = field(default=None)
"""

class OptimiseState1(State):

    def add_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # ターゲット行／サンプル行に分割
        df_target = df.filter(pl.col("tag") == "Target")
        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        df_train = df.filter(pl.col("tag") == "Sample")

        # サンプル行がなければ、元 df に Score=null を追加して返す
        if df_train.height == 0:
            return df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("Score")
            )

        # ターゲット値を結合してマンハッタン距離計算
        df_target_small = df_target.select(["well", "R", "G", "B"])
        print("target_well")
        print(df_target_small.select("well").unique().to_numpy())
        print("train_well")
        print(df_train.select("well").unique().to_numpy())
        df_join = (
            df_train
            .join(df_target_small, on="well", how="inner", suffix="_tgt")
            .drop_nulls(subset=[
                "R", "G", "B",
                "Color1_ratio", "Color2_ratio", "Color3_ratio"
            ])
            .with_columns(
                (
                    (pl.col("R_tgt") - pl.col("R")).abs()
                + (pl.col("G_tgt") - pl.col("G")).abs()
                + (pl.col("B_tgt") - pl.col("B")).abs()
                ).alias("Score")
            )
        )

        print("df_join")
        print(df_join)

        # df_join から "well","time","Score" を取り出し
        # （time 列が無ければキーを well のみで置き換えてください）
        scored_idx = df_join.select(["well", "Score"])
        print("scored_idx")
        print(scored_idx)

        # 元 df に left-join して、スコアがある行だけ Score を埋める
        df_out = df.join(scored_idx, on=["well"], how="left")

        return df_out

    def plot_data(self, df_train: pl.DataFrame, df_target: pl.DataFrame, df_next: pl.DataFrame, save_path: Path):
        """
        Args:
            df_train (pl.DataFrame): tag=="Sample" の行を含む DataFrame
            df_target (pl.DataFrame): tag=="Target" の行を含む DataFrame
            df_next (pl.DataFrame): 次の候補 1 行のみの DataFrame
            save_path (Path): 保存先パス
        """
        # ────────────── Step 0: add_score で Sample 行ごとの Score を計算 ──────────────
        # df_train と df_target を結合して add_score に渡す
        df_all   = df_train.vstack(df_target)
        df_scored = self.add_score(df_all)  # Sample 行に限って Score 列が追加される

        # プロット用のデータ抽出
        A        = df_scored["Color1_ratio"].to_numpy()
        B        = df_scored["Color2_ratio"].to_numpy()
        diff_sum = df_scored["Score"].to_numpy()  # 既に絶対値の和になっている

        # 次の候補点
        A_next = df_next["Color1_ratio"].to_numpy()
        B_next = df_next["Color2_ratio"].to_numpy()

        # プロット（2Dプロット A vs B）
        fig, ax = plt.subplots(figsize=(8, 6))

        # カラーマップを用いた散布図（Score が大きいほど色が濃く）
        sc = ax.scatter(A, B, c=diff_sum, cmap='viridis', marker='o', edgecolors='k', alpha=0.75)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Score (Manhattan distance)')

        # 次のデータポイントのプロット
        ax.scatter(A_next, B_next, facecolors="none", edgecolors='black',
                marker='o', s=100, label="Next")

        # ターゲット値のプロット（df_target の平均ではなく固定でも可）
        target_A = df_target["Color1_ratio"].mean()
        target_B = df_target["Color2_ratio"].mean()
        ax.scatter(target_A, target_B, color='red', s=100,
                edgecolors='black', label='Target', zorder=3)
        ax.text(target_A + 0.02, target_B, "Target",
                color='red', fontsize=12, verticalalignment='bottom')

        # 軸ラベル・範囲
        ax.set_xlabel('Color1 Ratio (A)')
        ax.set_ylabel('Color2 Ratio (B)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axline((0, 1), slope=-1, c="k")
        ax.legend()
        ax.grid(True)

        # 保存
        plt.savefig(save_path)


    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # ────────────── Step 1: tag による分割 ──────────────
        df_target = df.filter(pl.col("tag") == "Target")
        df_train  = df.filter(pl.col("tag") == "Sample")

        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        if df_train.height == 0:
            # サンプルがない場合はランダム出力
            candidates = torch.rand(8, 3, dtype=torch.double)
        else:
            # ────────────── Step 2: Score の計算 ──────────────
            # add_score を呼び出して、Sample 行ごとに L1 距離（Score）を計算
            df_join = self.add_score(df)
            print("df_join:")
            for row in df_join.rows():
                print(f"Row: {row}")

            df_target = df_join.filter(pl.col("tag") == "Target")
            df_train  = df_join.filter(pl.col("tag") == "Sample")

            # ────────────── Step 3: GP の入力 X と目的 Y の作成 ──────────────
            # 入力: 色水比率
            X_train = torch.tensor(
                df_train.select(["Color1_ratio", "Color2_ratio", "Color3_ratio"]).to_numpy(),
                dtype=torch.double
            )
            # Score は正の距離なので、目的関数はその負値
            Y = torch.tensor(
                (-df_train["Score"].to_numpy()).reshape(-1, 1),
                dtype=torch.double
            )

            # Show all the rows of X_train and Y without truncation
            print("X_train:")
            for i, row in enumerate(X_train):
                print(f"Row {i}: {row.tolist()}")
            print("Y:")
            for i, row in enumerate(Y):
                print(f"Row {i}: {row.item()}")
            # ────────────── Step 4: GP モデルの構築と学習 ──────────────
            model = SingleTaskGP(X_train, Y)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # ────────────── Step 5: Acquisition Function の設定 ──────────────
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_train,
            )

            # ────────────── Step 6: 最適化 ──────────────
            # 0.1 -1 の範囲で色水比率を最適化, lower bound 0.1, upper bound 1
            bounds = torch.stack([
                torch.full((3,), 0.1, dtype=torch.double),  # 下限を 0.1 に設定
                torch.full((3,), 0.8, dtype=torch.double)          # 上限は 1
            ])

            inequality_constraints = [
                (
                    torch.tensor([0,1,2], dtype=torch.long),
                    torch.tensor([-1.0,-1.0,-1.0], dtype=torch.double),
                    -1.0
                )
            ]
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=8,
                num_restarts=10,
                raw_samples=100,
                inequality_constraints=inequality_constraints,
                options={"batch_limit": 5, "maxiter": 200},
            )

        # ────────────── Step 7: 正規化と結果返却 ──────────────
        candidates_normalized = candidates / candidates.sum(dim=1, keepdim=True)
        result_df = pl.DataFrame({
            "Color1_ratio": candidates_normalized[:,0].tolist(),
            "Color2_ratio": candidates_normalized[:,1].tolist(),
            "Color3_ratio": candidates_normalized[:,2].tolist(),
        })

        # 以下は元コードのまま
        now_str        = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
        save_path      = Path(f"{now_str}__{uuid4()}_result.csv")
        plot_save_path = save_path.with_suffix(".png")
        result_df.write_csv(save_path)
        try:
            self.plot_data(df_train, df_target, result_df, plot_save_path)
        
        except Exception as e:
            print(f"Error while plotting data: {e}")    
        try:
            times = sorted(df["time"].to_list())
            current_time = times[-1]
        except:
            current_time = datetime.now().timestamp() // 60

        return TaskGroup(
            optimal_start_time=current_time,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=60, interval=0, experiment_operation=str(save_path), optimal_machine_type=0),
                Task(processing_time=5,  interval=0, experiment_operation="observation",       optimal_machine_type=1),
            ]
        )

    
    def transition_function(self, df: pl.DataFrame) -> str:
        # return the state name
        return "OptimiseState2"
        

class OptimiseState2(State):

    def add_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # ターゲット行／サンプル行に分割
        df_target = df.filter(pl.col("tag") == "Target")
        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        df_train = df.filter(pl.col("tag") == "Sample")

        # サンプル行がなければ、元 df に Score=null を追加して返す
        if df_train.height == 0:
            return df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("Score")
            )

        # ターゲット値を結合してマンハッタン距離計算
        df_target_small = df_target.select(["well", "R", "G", "B"])
        print("target_well")
        print(df_target_small.select("well").unique().to_numpy())
        print("train_well")
        print(df_train.select("well").unique().to_numpy())
        df_join = (
            df_train
            .join(df_target_small, on="well", how="inner", suffix="_tgt")
            .drop_nulls(subset=[
                "R", "G", "B",
                "Color1_ratio", "Color2_ratio", "Color3_ratio"
            ])
            .with_columns(
                (
                    (pl.col("R_tgt") - pl.col("R")).abs()
                + (pl.col("G_tgt") - pl.col("G")).abs()
                + (pl.col("B_tgt") - pl.col("B")).abs()
                ).alias("Score")
            )
        )

        print("df_join")
        print(df_join)

        # df_join から "well","time","Score" を取り出し
        # （time 列が無ければキーを well のみで置き換えてください）
        scored_idx = df_join.select(["well", "Score"])
        print("scored_idx")
        print(scored_idx)

        # 元 df に left-join して、スコアがある行だけ Score を埋める
        df_out = df.join(scored_idx, on=["well"], how="left")

        return df_out

    def plot_data(self, df_train: pl.DataFrame, df_target: pl.DataFrame, df_next: pl.DataFrame, save_path: Path):
        """
        Args:
            df_train (pl.DataFrame): tag=="Sample" の行を含む DataFrame
            df_target (pl.DataFrame): tag=="Target" の行を含む DataFrame
            df_next (pl.DataFrame): 次の候補 1 行のみの DataFrame
            save_path (Path): 保存先パス
        """
        # ────────────── Step 0: add_score で Sample 行ごとの Score を計算 ──────────────
        # df_train と df_target を結合して add_score に渡す
        df_all   = df_train.vstack(df_target)
        df_scored = self.add_score(df_all)  # Sample 行に限って Score 列が追加される

        # プロット用のデータ抽出
        A        = df_scored["Color1_ratio"].to_numpy()
        B        = df_scored["Color2_ratio"].to_numpy()
        diff_sum = df_scored["Score"].to_numpy()  # 既に絶対値の和になっている

        # 次の候補点
        A_next = df_next["Color1_ratio"].to_numpy()
        B_next = df_next["Color2_ratio"].to_numpy()

        # プロット（2Dプロット A vs B）
        fig, ax = plt.subplots(figsize=(8, 6))

        # カラーマップを用いた散布図（Score が大きいほど色が濃く）
        sc = ax.scatter(A, B, c=diff_sum, cmap='viridis', marker='o', edgecolors='k', alpha=0.75)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Score (Manhattan distance)')

        # 次のデータポイントのプロット
        ax.scatter(A_next, B_next, facecolors="none", edgecolors='black',
                marker='o', s=100, label="Next")

        # ターゲット値のプロット（df_target の平均ではなく固定でも可）
        target_A = df_target["Color1_ratio"].mean()
        target_B = df_target["Color2_ratio"].mean()
        ax.scatter(target_A, target_B, color='red', s=100,
                edgecolors='black', label='Target', zorder=3)
        ax.text(target_A + 0.02, target_B, "Target",
                color='red', fontsize=12, verticalalignment='bottom')

        # 軸ラベル・範囲
        ax.set_xlabel('Color1 Ratio (A)')
        ax.set_ylabel('Color2 Ratio (B)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axline((0, 1), slope=-1, c="k")
        ax.legend()
        ax.grid(True)

        # 保存
        plt.savefig(save_path)

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # ────────────── Step 1: tag による分割 ──────────────
        df_target = df.filter(pl.col("tag") == "Target")
        df_train  = df.filter(pl.col("tag") == "Sample")

        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        if df_train.height == 0:
            # サンプルがない場合はランダム出力
            candidates = torch.rand(8, 3, dtype=torch.double)
        else:
            # ────────────── Step 2: Score の計算 ──────────────
            # add_score を呼び出して、Sample 行ごとに L1 距離（Score）を計算
            df_join = self.add_score(df)
            print("df_join:")
            for row in df_join.rows():
                print(f"Row: {row}")

            df_target = df_join.filter(pl.col("tag") == "Target")
            df_train  = df_join.filter(pl.col("tag") == "Sample")

            # ────────────── Step 3: GP の入力 X と目的 Y の作成 ──────────────
            # 入力: 色水比率
            X_train = torch.tensor(
                df_train.select(["Color1_ratio", "Color2_ratio", "Color3_ratio"]).to_numpy(),
                dtype=torch.double
            )
            # Score は正の距離なので、目的関数はその負値
            Y = torch.tensor(
                (-df_train["Score"].to_numpy()).reshape(-1, 1),
                dtype=torch.double
            )

            # Show all the rows of X_train and Y without truncation
            print("X_train:")
            for i, row in enumerate(X_train):
                print(f"Row {i}: {row.tolist()}")
            print("Y:")
            for i, row in enumerate(Y):
                print(f"Row {i}: {row.item()}")
            # ────────────── Step 4: GP モデルの構築と学習 ──────────────
            model = SingleTaskGP(X_train, Y)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # ────────────── Step 5: Acquisition Function の設定 ──────────────
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_train,
            )

            # ────────────── Step 6: 最適化 ──────────────
            bounds = torch.stack([
                torch.full((3,), 0.1, dtype=torch.double),  # 下限を 0.1 に設定
                torch.full((3,), 0.8, dtype=torch.double)          # 上限は 1
            ])
            inequality_constraints = [
                (
                    torch.tensor([0,1,2], dtype=torch.long),
                    torch.tensor([-1.0,-1.0,-1.0], dtype=torch.double),
                    -1.0
                )
            ]
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=8,
                num_restarts=10,
                raw_samples=100,
                inequality_constraints=inequality_constraints,
                options={"batch_limit": 5, "maxiter": 200},
            )

        # ────────────── Step 7: 正規化と結果返却 ──────────────
        candidates_normalized = candidates / candidates.sum(dim=1, keepdim=True)
        result_df = pl.DataFrame({
            "Color1_ratio": candidates_normalized[:,0].tolist(),
            "Color2_ratio": candidates_normalized[:,1].tolist(),
            "Color3_ratio": candidates_normalized[:,2].tolist(),
        })

        # 以下は元コードのまま
        now_str        = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
        save_path      = Path(f"{now_str}__{uuid4()}_result.csv")
        plot_save_path = save_path.with_suffix(".png")
        result_df.write_csv(save_path)
        try:
            self.plot_data(df_train, df_target, result_df, plot_save_path)
        
        except Exception as e:
            print(f"Error while plotting data: {e}")    
        try:
            times = sorted(df["time"].to_list())
            current_time = times[-1]
        except:
            current_time = datetime.now().timestamp() // 60

        return TaskGroup(
            optimal_start_time=current_time,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=60, interval=0, experiment_operation=str(save_path), optimal_machine_type=0),
                Task(processing_time=5,  interval=0, experiment_operation="observation",       optimal_machine_type=1),
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        # return the state name
        return "OptimiseState3"
    

class OptimiseState3(State):

    def add_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # ターゲット行／サンプル行に分割
        df_target = df.filter(pl.col("tag") == "Target")
        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        df_train = df.filter(pl.col("tag") == "Sample")

        # サンプル行がなければ、元 df に Score=null を追加して返す
        if df_train.height == 0:
            return df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("Score")
            )

        # ターゲット値を結合してマンハッタン距離計算
        df_target_small = df_target.select(["well", "R", "G", "B"])
        print("target_well")
        print(df_target_small.select("well").unique().to_numpy())
        print("train_well")
        print(df_train.select("well").unique().to_numpy())
        df_join = (
            df_train
            .join(df_target_small, on="well", how="inner", suffix="_tgt")
            .drop_nulls(subset=[
                "R", "G", "B",
                "Color1_ratio", "Color2_ratio", "Color3_ratio"
            ])
            .with_columns(
                (
                    (pl.col("R_tgt") - pl.col("R")).abs()
                + (pl.col("G_tgt") - pl.col("G")).abs()
                + (pl.col("B_tgt") - pl.col("B")).abs()
                ).alias("Score")
            )
        )

        print("df_join")
        print(df_join)

        # df_join から "well","time","Score" を取り出し
        # （time 列が無ければキーを well のみで置き換えてください）
        scored_idx = df_join.select(["well", "Score"])
        print("scored_idx")
        print(scored_idx)

        # 元 df に left-join して、スコアがある行だけ Score を埋める
        df_out = df.join(scored_idx, on=["well"], how="left")

        return df_out

    def plot_data(self, df_train: pl.DataFrame, df_target: pl.DataFrame, df_next: pl.DataFrame, save_path: Path):
        """
        Args:
            df_train (pl.DataFrame): tag=="Sample" の行を含む DataFrame
            df_target (pl.DataFrame): tag=="Target" の行を含む DataFrame
            df_next (pl.DataFrame): 次の候補 1 行のみの DataFrame
            save_path (Path): 保存先パス
        """
        # ────────────── Step 0: add_score で Sample 行ごとの Score を計算 ──────────────
        # df_train と df_target を結合して add_score に渡す
        df_all   = df_train.vstack(df_target)
        df_scored = self.add_score(df_all)  # Sample 行に限って Score 列が追加される

        # プロット用のデータ抽出
        A        = df_scored["Color1_ratio"].to_numpy()
        B        = df_scored["Color2_ratio"].to_numpy()
        diff_sum = df_scored["Score"].to_numpy()  # 既に絶対値の和になっている

        # 次の候補点
        A_next = df_next["Color1_ratio"].to_numpy()
        B_next = df_next["Color2_ratio"].to_numpy()

        # プロット（2Dプロット A vs B）
        fig, ax = plt.subplots(figsize=(8, 6))

        # カラーマップを用いた散布図（Score が大きいほど色が濃く）
        sc = ax.scatter(A, B, c=diff_sum, cmap='viridis', marker='o', edgecolors='k', alpha=0.75)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Score (Manhattan distance)')

        # 次のデータポイントのプロット
        ax.scatter(A_next, B_next, facecolors="none", edgecolors='black',
                marker='o', s=100, label="Next")

        # ターゲット値のプロット（df_target の平均ではなく固定でも可）
        target_A = df_target["Color1_ratio"].mean()
        target_B = df_target["Color2_ratio"].mean()
        ax.scatter(target_A, target_B, color='red', s=100,
                edgecolors='black', label='Target', zorder=3)
        ax.text(target_A + 0.02, target_B, "Target",
                color='red', fontsize=12, verticalalignment='bottom')

        # 軸ラベル・範囲
        ax.set_xlabel('Color1 Ratio (A)')
        ax.set_ylabel('Color2 Ratio (B)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axline((0, 1), slope=-1, c="k")
        ax.legend()
        ax.grid(True)

        # 保存
        plt.savefig(save_path)

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # ────────────── Step 1: tag による分割 ──────────────
        df_target = df.filter(pl.col("tag") == "Target")
        df_train  = df.filter(pl.col("tag") == "Sample")

        if df_target.height == 0:
            raise ValueError("df に tag=='Target' の行が見つかりません。")
        if df_train.height == 0:
            # サンプルがない場合はランダム出力
            candidates = torch.rand(8, 3, dtype=torch.double)
        else:
            # ────────────── Step 2: Score の計算 ──────────────
            # add_score を呼び出して、Sample 行ごとに L1 距離（Score）を計算
            df_join = self.add_score(df)
            print("df_join:")
            for row in df_join.rows():
                print(f"Row: {row}")

            df_target = df_join.filter(pl.col("tag") == "Target")
            df_train  = df_join.filter(pl.col("tag") == "Sample")

            # ────────────── Step 3: GP の入力 X と目的 Y の作成 ──────────────
            # 入力: 色水比率
            X_train = torch.tensor(
                df_train.select(["Color1_ratio", "Color2_ratio", "Color3_ratio"]).to_numpy(),
                dtype=torch.double
            )
            # Score は正の距離なので、目的関数はその負値
            Y = torch.tensor(
                (-df_train["Score"].to_numpy()).reshape(-1, 1),
                dtype=torch.double
            )

            # Show all the rows of X_train and Y without truncation
            print("X_train:")
            for i, row in enumerate(X_train):
                print(f"Row {i}: {row.tolist()}")
            print("Y:")
            for i, row in enumerate(Y):
                print(f"Row {i}: {row.item()}")

            # ────────────── Step 4: GP モデルの構築と学習 ──────────────
            model = SingleTaskGP(X_train, Y)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # ────────────── Step 5: Acquisition Function の設定 ──────────────
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_train,
            )

            # ────────────── Step 6: 最適化 ──────────────
            bounds = torch.stack([
                torch.full((3,), 0.1, dtype=torch.double),  # 下限を 0.1 に設定
                torch.full((3,), 0.8, dtype=torch.double)          # 上限は 1
            ])
            inequality_constraints = [
                (
                    torch.tensor([0,1,2], dtype=torch.long),
                    torch.tensor([-1.0,-1.0,-1.0], dtype=torch.double),
                    -1.0
                )
            ]
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=8,
                num_restarts=10,
                raw_samples=100,
                inequality_constraints=inequality_constraints,
                options={"batch_limit": 5, "maxiter": 200},
            )

        # ────────────── Step 7: 正規化と結果返却 ──────────────
        candidates_normalized = candidates / candidates.sum(dim=1, keepdim=True)
        result_df = pl.DataFrame({
            "Color1_ratio": candidates_normalized[:,0].tolist(),
            "Color2_ratio": candidates_normalized[:,1].tolist(),
            "Color3_ratio": candidates_normalized[:,2].tolist(),
        })

        # 以下は元コードのまま
        now_str        = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
        save_path      = Path(f"{now_str}__{uuid4()}_result.csv")
        plot_save_path = save_path.with_suffix(".png")
        result_df.write_csv(save_path)
        try:
            self.plot_data(df_train, df_target, result_df, plot_save_path)
        
        except Exception as e:
            print(f"Error while plotting data: {e}")    
        try:
            times = sorted(df["time"].to_list())
            current_time = times[-1]
        except:
            current_time = datetime.now().timestamp() // 60

        return TaskGroup(
            optimal_start_time=current_time,
            penalty_type=NonePenalty(),
            tasks=[
                Task(processing_time=60, interval=0, experiment_operation=str(save_path), optimal_machine_type=0),
                Task(processing_time=5,  interval=0, experiment_operation="observation",       optimal_machine_type=1),
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        # return the state name
        return "ExpireState"
        
class ExpireState(State):

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time = int(UNIX_2038_01_01_00_00_00_minutes)
        return TaskGroup(
            optimal_start_time=optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=5,
                    interval=0, 
                    experiment_operation="Expire",
                    optimal_machine_type = 1,
                    )
            ]
        )
        

    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"
    
    

def gen_ot2_cwo_experiment(experiment_name = "gen_ot2_cwo_experiment") -> Experiment:
    return Experiment(
        experiment_name=experiment_name,
        states=[
            OptimiseState1(),
            OptimiseState2(),
            OptimiseState3(),
            ExpireState()
        ],
        current_state_name="OptimiseState1",
        # 405nm,450nm,492nm,620nm,Color1_ratio,Color2_ratio,Color3_ratio,time,well
        shared_variable_history=
        pl.read_csv(TARGET_DATA_PATH)

    )

def gen_ot2_cwo_experiments(temp_dir: str, experiment_name = "standard_experiment") -> Experiments:
    # Path is volatile
    lab = Experiments(
        experiments=[
            gen_ot2_cwo_experiment(experiment_name=experiment_name)
        ],
        parent_dir_path = Path(temp_dir)
    )

    lab.machine_list.add_machine(Machine(machine_type = 0, description = "OT-2"))
    lab.machine_list.add_machine(Machine(machine_type = 1, description = "Human"))
    return lab