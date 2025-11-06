from pathlib import Path
import polars as pl

result_csv = Path("/Users/yuyaarai/Documents/Humanics/Project/gems_ot2_colour-water-optimisation-simulator/ot2_experiment/step_00000023/experiments/gen_ot2_cwo_experiment_46fdf7bf-3fa8-4ce0-9748-8f56cee29ade_shared_variable_history.csv")

df = pl.read_csv(result_csv)
print(df)

df_target = df.filter(pl.col("well") == "Target")
target_R = df_target.select("R")[0]
target_G = df_target.select("G")[0]
target_B = df_target.select("B")[0]
df_train = df.filter(pl.col("well") != "Target")

df_train = df_train.drop_nulls(subset=["R", "G", "B"])

# Sort the dataframe by the distance from the target well
# In other words, min |R_target - R| + |G_target - G| + |B_target - B|
df_train = df_train.with_columns(
    ((pl.col("R") - target_R)**2 +
    (pl.col("G") - target_G +
    (pl.col("B") - target_B)**2))
    .alias("distance")
)

df_train = df_train.sort("distance")
df_train.write_csv("sorted.csv")
print(df_train)

