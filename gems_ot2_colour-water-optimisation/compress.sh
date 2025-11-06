#!/usr/bin/env bash

# ffmpeg 圧縮オプション
CRF=31

# ファイル名パターン
PATTERN='*_hh_mm_ss_experiment_movie.mp4'

# -- 処理本体 --
echo "対象ファイルを検索します: $PATTERN"
files=( $PATTERN )

if [ ${#files[@]} -eq 0 ]; then
  echo "マッチするファイルがありません。"
  exit 0
fi

for src in "${files[@]}"; do
  # 圧縮後ファイル名
  dst="${src%.mp4}_compressed.mp4"

  # 既に圧縮後ファイルが存在するならスキップ
  if [ -f "$dst" ]; then
    echo "スキップ（既に存在）: $dst"
    continue
  fi

  echo "圧縮中: $src → $dst"
  ffmpeg -i "$src" -crf "$CRF" "$dst"

  echo "  git add \"$dst\""
  git add "$dst"
done