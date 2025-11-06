import argparse
import matplotlib.pyplot as plt  # 画像を表示するためのモジュール
from take_picture import take_picture
import cv2  # OpenCVをインポート
import numpy as np  # numpyをインポート
import string
import pandas as pd

ROW_NUM = 8
COL_NUM = 12
AVE_CALC_SIZE = 3
CROSS_LEN = AVE_CALC_SIZE 
# TODO: Change the reference points to the actual reference points based on the camera position
REFERENCE_POINTS = [(854, 520), (1095, 516), (1098, 670)]

def create_cross(img_array, centre: list = (10, 10), xy_len:int = 40):
    # xy_len = min(len(img_array), len(img_array[0])) // 10
    draw_width = xy_len//2
    for x in range(-xy_len, xy_len):
        x = x + centre[0]
        for width in range(-draw_width // 2, draw_width // 2):
            y = centre[1] + width
            if 0 <= x < len(img_array) and 0 <= y < len(img_array[0]):
                    img_array[x][y] = [255, 255, 0]
    for y in range(-xy_len, xy_len):
        x = centre[0]
        y = y + centre[1]
        for width in range(-draw_width // 2, draw_width // 2):
            x = centre[0] + width
            if 0 <= x < len(img_array) and 0 <= y < len(img_array[0]):
                    img_array[x][y] = [255, 255, 0]
    return img_array

def create_box(img_array, centre: list = (10, 10), xy_len:int = 40):
    # xy_len = min(len(img_array), len(img_array[0])) // 10
    for x_ in range(-xy_len, xy_len):
        for y_ in range(-xy_len, xy_len):
            x = x_ + centre[0]
            y = y_ + centre[1]
            if 0 <= x < len(img_array) and 0 <= y < len(img_array[0]):
                img_array[x][y] = [255, 255, 0]
    return img_array

def create_parallelogram(A: list, B: list, C: list):
    return (C[0] - (B[0] - A[0]), C[1] - (B[1] - A[1]))

def calc_well_places(A, B, C, row_num, col_num):
    D = create_parallelogram(A, B, C)
    wells = dict()
    for i, ch in enumerate(string.ascii_uppercase[:row_num]):
        for j in range(col_num):
            x = A[0] + (B[0] - A[0]) * j / (col_num - 1) + (D[0] - A[0]) * i / (row_num - 1)
            y = A[1] + (B[1] - A[1]) * j / (col_num - 1) + (D[1] - A[1]) * i / (row_num - 1)
            wells[f"{ch}{j+1}"] = (int(x), int(y))
    return wells

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="96-well plate detection from an image")
    parser.add_argument("-o", "--out_dir", default="RGB_converter/test", type=str, help="Path to save the output file", required=False)
    parser.add_argument("-s","--start_column", default=1, type=int, help="Starting column number", required=False)
    parser.add_argument("-r", "--ratios", default="samples/ratio.csv", type=str, help="Path to the ratios file", required=False)
    parser.add_argument("-t", "--tag", default="Sample", type=str, help="Tag for the output files", required=False)
    args = parser.parse_args()
    print(f"{args=}")
    out_dir = Path(args.out_dir)
    save_path = out_dir / "well_plate.jpg"
    img_path = take_picture(retry_num=5, focus_time=5.0, save_path=save_path)
    print(f"{img_path=}")
        
    img = cv2.imread(img_path)  # 画像の読み込み
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 色配置の変換 BGR→RGB
    img_array = np.asarray(img)  # numpyで扱える配列をつくる
    # plt.imshow(img_array)  # 画像の表示
    # plt.show()

    A, B, C = REFERENCE_POINTS
    A = (A[1], A[0])
    B = (B[1], B[0])
    C = (C[1], C[0])

    wells = calc_well_places(A, B, C, ROW_NUM, COL_NUM)
    colours = dict()
    r = AVE_CALC_SIZE
    for key, well in wells.items():
        # Get average colour of the well
        x, y = well
        well_img = img_array[x-r:x+r, y-r:y+r]
        avg_colour = np.median(well_img, axis=(0, 1))
        colours[key] = avg_colour
    for c in colours.values():
        print(c)
    df = pd.DataFrame(list(colours.items()), columns=["well", "Colour"])
    df["R"] = df["Colour"].apply(lambda x: x[2])
    df["G"] = df["Colour"].apply(lambda x: x[1])
    df["B"] = df["Colour"].apply(lambda x: x[0])

    df.drop("Colour", axis=1, inplace=True)
    df.to_csv(out_dir/"well_colours.csv", index=False)

    # Slice the df according to the start_column number (df[well] num = column, column + 1, column + 2, column + 3)

    df_slice = df[df["well"].apply(lambda x: int(x[1:]) >= args.start_column and int(x[1:]) < args.start_column + 4)]
    df_slice.to_csv(out_dir/"sliced_well_colours.csv", index=False)




    for well in wells.values():
        img_array = create_box(img_array, well, xy_len=CROSS_LEN)

    cv2.imwrite(str(out_dir / "well_plate.png"), img_array)

    colour_df = df_slice
    colour_df["column"] = colour_df["well"].apply(lambda x: x[1:])
    colour_df["row"] = colour_df["well"].apply(lambda x: x[0])
    #   well     R      G     B
    #   A5  75.5  102.5  59.5
    ratios = pd.read_csv(args.ratios)
    ratios["row"] = "A"
    for i, ratio_set in enumerate(ratios.values):
        row_alphabet = string.ascii_uppercase[i]
        ratios.loc[i, "row"] = row_alphabet
    #        Color1_ratio  Color2_ratio  Color3_ratio
    #        0.140715      0.793967      0.065319

    merged_df = pd.merge(colour_df, ratios, on="row")
    
    # Add tag to the merged_df
    merged_df["tag"] = args.tag

    print(merged_df)

    merged_df.to_csv(Path(out_dir)/"merged.csv", index=False)


