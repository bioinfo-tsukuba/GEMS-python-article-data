import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_well_colors(csv_path: str):
    # CSV の読み込み（ファイル名を適宜変更）
    df = pd.read_csv(csv_path)

    # 8行（A～H）、12列の空の配列（各セルは RGB 値）を用意（matplotlib は [0,1] の範囲の色を使用）
    grid = np.zeros((8, 12, 3), dtype=float)

    # CSV の各行のウェル名から、グリッド上の位置（行, 列）を決定し、RGB 色をセット
    for index, row in df.iterrows():
        well = row['well']
        # ウェル名は "A1", "B12" のような形式と仮定
        row_letter = well[0]              # 例："A"
        col_number = int(well[1:]) - 1      # 数字部分（1-indexedなので 1 を引く）
        row_index = ord(row_letter.upper()) - ord('A')  # A→0, B→1, ... H→7

        # 例示されている値は 0～255 の範囲と仮定し、[0,1] に正規化
        R = row['R'] / 255.0
        G = row['G'] / 255.0
        B = row['B'] / 255.0

        grid[row_index, col_number, :] = [R, G, B]

    # 描画用の図を作成
    fig, ax = plt.subplots(figsize=(12, 8))
    # グリッドを画像として表示（各セルが対応する色で塗りつぶされる）
    ax.imshow(grid, aspect='auto')

    # 各セル中央にウェル名を表示
    for i in range(8):        # 行 A～H
        for j in range(12):   # 列 1～12
            well_name = chr(ord('A') + i) + str(j + 1)
            # セルの平均明度に応じて文字色を選ぶ（背景が暗いなら白、明るいなら黒）
            avg_brightness = np.mean(grid[i, j, :])
            text_color = 'white' if avg_brightness < 0.5 else 'black'
            ax.text(j, i, well_name, ha='center', va='center', color=text_color, fontsize=10)

            # セルの中央に RGB 値を表示
            R, G, B = grid[i, j, :]
            ax.text(j, i + 0.1, f"({R:.2f}, {G:.2f}, {B:.2f})", ha='center', va='center', color=text_color, fontsize=8)

    # 軸の設定：x 軸は 1～12、y 軸は A～H
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels([chr(ord('A') + i) for i in range(8)])

    # 余計な軸の枠や目盛りを非表示にする場合
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # CSV ファイルのパスを指定して可視化
    visualize_well_colors(input("Enter the path to the CSV file: "))