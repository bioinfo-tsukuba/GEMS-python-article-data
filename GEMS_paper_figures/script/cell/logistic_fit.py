# GEMS_paper_figures/script/cell/curve_estimatitor_summarise.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from typing import List, Tuple, Optional, Union

import numpy as np
import polars as pl
from scipy.optimize import curve_fit


# -----------------------------
# 基本モデル（K は固定）
# -----------------------------
def logistic(t: np.ndarray, r: float, n0: float, K: float = 1.0) -> np.ndarray:
    """
    ロジスティック増殖モデル: n(t) = K / (1 + (K/n0 - 1) * exp(-r t))
    """
    return K / (1 + (K / n0 - 1.0) * np.exp(-r * t))


# -----------------------------
# Passage グルーピング
# -----------------------------
def group_by_passage(
    df: pl.DataFrame,
    time_column: str = "time",
    operation_column: str = "operation",
    passage_operation_name: str = "Passage",
) -> pl.DataFrame:
    """
    Passage ごとにグループ番号 (passage_group) を振る。
    - Passage が無い場合は全行を 1 グループ (0) として扱う。
    - 最初の Passage 以降の行のみを対象にする。
    """
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."

    df = df.sort(time_column)

    # Passage が全く無い場合は全行を 1 グループ扱い
    passage_df = df.filter(pl.col(operation_column) == passage_operation_name)
    if passage_df.height == 0:
        print("First Passage Time not found -> treat all rows as a single passage_group=0")
        return df.with_columns(pl.lit(0).alias("passage_group"))

    # 最初の Passage 以降に限定
    first_passage_time = passage_df.select(time_column).head(1)[time_column].item()
    df = df.filter(pl.col(time_column) >= first_passage_time)

    # Passage の時刻リスト
    passage_times = (
        df.filter(pl.col(operation_column) == passage_operation_name)[time_column].to_list()
    )

    # passage_group を付与（次の Passage が始まるまで同じグループ）
    return df.with_columns(
        pl.Series("passage_group", np.searchsorted(passage_times[1:], df[time_column], side="right"))
    )


# -----------------------------
# フィット & 保存（JSON / CSV）
# -----------------------------
def fit_param_with_weight(
    df: pl.DataFrame,
    time_column: str = "time",
    density_column: str = "density",
    operation_column: str = "operation",
    density_operation_name: str = "GetImage",
    passage_operation_name: str = "Passage",
    save_path: Optional[Union[str, Path]] = None,
    constant_K: float = 1.0,  # K を固定
) -> Tuple[float, float, List[float], pl.DataFrame]:
    """
    各 passage_group の初期密度 n0 を同時推定しつつ、全体の r（増殖率）を推定する。
    K は constant_K に固定してフィットしない。

    save_path が与えられた場合は、以下を保存する:
      - {save_path}.json : {"k": constant_K, "r": r, "n0": [ ... ]}  # 互換のためキー名 "k" を継続
      - {save_path}.csv  : passage_group 付与済みの元データ（グルーピング後）

    戻り値: (K, r, n0_list, df_grouped)
    """
    # 必須列の存在確認
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert density_column in df.columns, f"Column {density_column} as density column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."

    # passage_group の付与
    df_grouped = df
    if "passage_group" not in df_grouped.columns:
        df_grouped = group_by_passage(
            df=df,
            time_column=time_column,
            operation_column=operation_column,
            passage_operation_name=passage_operation_name,
        )

    # フィット対象: Passage 行と密度観測行のみ
    df_fit = df_grouped.filter(
        (pl.col(operation_column) == density_operation_name)
        | (pl.col(operation_column) == passage_operation_name)
    ).select([time_column, density_column, operation_column, "passage_group"])

    if df_fit.height == 0:
        raise ValueError("No rows to fit after filtering by operation names.")

    # グループ数
    group_count = int(df_fit["passage_group"].max()) + 1

    # 各グループの n0 初期推定
    min_densities_by_group = (
        df_fit.filter(pl.col(density_column).is_not_null())
        .group_by("passage_group")
        .agg(pl.min(density_column).alias("n0"))
        .to_dict(as_series=False)
    )
    n0_map = dict(zip(min_densities_by_group.get("passage_group", []), min_densities_by_group.get("n0", [])))

    # 全体最小値（フォールバック）
    overall_min = (
        df_fit.filter(pl.col(density_column).is_not_null())
        .select(pl.min(density_column))
        .to_series()
        .item()
    )
    if overall_min is None or overall_min <= 0:
        overall_min = 0.01  # 0 は不可（対数の都合）

    n0_estimates: List[float] = [float(n0_map.get(g, overall_min)) for g in range(group_count)]

    # curve_fit の重み（古いグループを相対的に軽く）
    groups = df_fit["passage_group"].to_numpy()
    sigma = (0.5) ** groups  # sigma が小さいほど重みが大きくなる

    # curve_fit 用配列
    t_arr = df_fit[time_column].to_numpy().astype(float)
    y_arr = df_fit[density_column].fill_null(0.0).to_numpy().astype(float)
    pg_arr = df_fit["passage_group"].to_numpy()

    # K 固定の結合モデル（r と 各グループの n0 を推定）
    def combined_model(t: np.ndarray, r: float, *n0_values: float) -> np.ndarray:
        """
        各グループの最初の時刻を基準（t=0）にしてロジスティックを生成。
        Passage 行（基準行）の観測値は 0 と見なす。
        """
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(t, dtype=float)

        for g in range(group_count):
            mask = (pg_arr == g)
            if not np.any(mask):
                continue

            times = t[mask].copy()
            times.sort()
            base_time = times[0]

            # 基準インデックス（当該グループの最初の行）
            base_idx_candidates = np.where((t == base_time) & mask)[0]
            base_idx = int(base_idx_candidates[0]) if base_idx_candidates.size > 0 else np.where(t == base_time)[0][0]

            rel_times = t[mask] - base_time
            n0 = float(n0_values[g]) if g < len(n0_values) else overall_min

            # K は固定
            result[mask] = logistic(rel_times, r, n0, K=constant_K)
            result[base_idx] = 0.0  # Passage 行は 0 として扱う

        return result

    # 初期値（r, n0_0..n0_{G-1}）
    initial_guess = [1e-5] + n0_estimates

    popt, _ = curve_fit(
        combined_model,
        t_arr,
        y_arr,
        p0=initial_guess,
        sigma=sigma,
        absolute_sigma=False,
        maxfev=200000,
    )

    r = float(popt[0])
    n0_fitted = [float(x) for x in popt[1:]]

    assert len(n0_fitted) == group_count, (
        f"Number of n0_fitted ({len(n0_fitted)}) is not equal to the number of passage groups ({group_count})."
    )

    # ---- 保存処理 ----
    if save_path is not None:
        try:
            base = Path(save_path)
            base.parent.mkdir(parents=True, exist_ok=True)

            json_path = base.with_suffix(".json")
            csv_path = base.with_suffix(".csv")

            # 互換のため "k" キーに固定値 K を入れる
            payload = {
                "k": constant_K,   # ここは固定 K（互換目的）
                "r": r,
                "n0": n0_fitted,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            df_grouped.write_csv(csv_path)

            print(f"[saved] {json_path}")
            print(f"[saved] {csv_path}")
        except Exception as e:
            print(f"Error saving the parameters: {e}")

    # 返り値は (K, r, n0_list, df_grouped) とする（下位互換）
    return constant_K, r, n0_fitted, df_grouped


# -----------------------------
# 実行部（可視化は行わない）
# -----------------------------
if __name__ == "__main__":
    # 入出力設定
    dir_path = Path("../GEMS-python-cell-culture/iPSsimulation/results/2024-11-13_real_round3/step_current/experiments")
    output_path = Path("data/241106_CCDS_253g1-hek293a_report/processed_data/curve_estimator_summary")
    output_path.mkdir(parents=True, exist_ok=True)

    # 列名
    time_column = "time"
    density_column = "density"
    operation_column = "operation"

    # オペレーション名（データの命名に合わせて変更可）
    HEK_density_operation_name = "HEKGetImage"
    HEK_passage_operation_name = "HEKPassage"
    iPS_density_operation_name = "iPSGetImage"
    iPS_passage_operation_name = "iPSPassage"

    # K を固定
    FIXED_K = 1.0

    # 対象セルタイプごとに処理
    for cell_type in ["HEK", "iPS"]:
        print(f"Cell Type: {cell_type}")
        try:
            for file in dir_path.iterdir():
                if not (file.is_file() and file.suffix == ".csv" and cell_type in file.stem):
                    continue

                print(f"Reading {file}")
                df = pl.read_csv(file)

                # セルタイプに応じて operation 名を切り替え
                if cell_type == "HEK":
                    density_op = HEK_density_operation_name
                    passage_op = HEK_passage_operation_name
                else:
                    density_op = iPS_density_operation_name
                    passage_op = iPS_passage_operation_name

                # 保存先のベースパス（.json / .csv は関数側で付与）
                save_base = output_path / f"{file.stem}_fit"

                try:
                    K_used, r, n0_list, df_grouped = fit_param_with_weight(
                        df=df,
                        time_column=time_column,
                        density_column=density_column,
                        operation_column=operation_column,
                        density_operation_name=density_op,
                        passage_operation_name=passage_op,
                        save_path=save_base,
                        constant_K=FIXED_K,  # K を固定
                    )
                    print(f"{file.name}: K={K_used}, r={r}, n0={n0_list}")
                except Exception as e:
                    print(f"Fit error for {file.name}: {e}")
        except Exception as e:
            print(f"Error in processing {cell_type}: {e}")
