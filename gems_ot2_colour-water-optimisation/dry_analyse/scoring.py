import polars as pl
from pathlib import Path

def add_score(df: pl.DataFrame) -> pl.DataFrame:
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

if __name__ == "__main__":
    path = Path("ot2_experiment/step_current/experiments/gen_ot2_cwo_experiment_aa9a0574-ca55-4813-9d25-2a66161e4afd_shared_variable_history.csv")
    df = pl.read_csv(path)

    print("元のデータフレーム:")
    print(df.head())

    df_scored = add_score(df)
    output_path = path.with_name("scored.csv")
    # This folder, dynamically get the path to the output folder
    folder = Path(__file__).parent
    output_path = folder / output_path.name
    df_scored.write_csv(output_path)
    print(f"スコアを追加したデータフレームを {output_path} に保存しました。")