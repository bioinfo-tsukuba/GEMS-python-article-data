import polars as pl

# CSVファイルの読み込み
file_path = "artifacts/CW/scored.csv"
df = pl.read_csv(file_path)

# データフレームのスキーマと先頭数行を表示
print(df.schema)
print(df.head())

# 'well'列が欠損しておらず、値が 'Target' でない行を抽出
df_filtered = df.filter(
    pl.col("tag").is_not_null() & (pl.col("tag") != "Target")
)

# 'column'の値に応じたROUND列を追加
df_filtered = df_filtered.with_columns(
    pl.when((pl.col("column") >= 1) & (pl.col("column") <= 4)).then(1)
      .when((pl.col("column") >= 5) & (pl.col("column") <= 8)).then(2)
      .when((pl.col("column") >= 9) & (pl.col("column") <= 12)).then(3)
      .otherwise(None)
      .alias("ROUND")
)

# 結果を新しいCSVファイルに保存
output_path = "artifacts/CW/filtered_with_round_scored.csv"
df_filtered.write_csv(output_path)

print(output_path)
