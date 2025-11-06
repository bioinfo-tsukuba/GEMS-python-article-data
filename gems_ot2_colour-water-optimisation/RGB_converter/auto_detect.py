import argparse
import matplotlib.pyplot as plt  # 画像を表示するためのモジュール
import cv2  # OpenCVをインポート
import numpy as np  # numpyをインポート
import string
import pandas as pd

ROW_NUM = 8
COL_NUM = 12

def create_cross(img_array, centre: list = (10, 10), xy_len:int = 40):
    # xy_len = min(len(img_array), len(img_array[0])) // 10
    draw_width = xy_len // 10
    for x in range(-xy_len, xy_len):
        x = x + centre[0]
        for width in range(-draw_width // 2, draw_width // 2):
            y = centre[1] + width
            if 0 <= x < len(img_array) and 0 <= y < len(img_array[0]):
                if (x, y) != centre:
                    img_array[x][y] = [255, 255, 0]
    for y in range(-xy_len, xy_len):
        x = centre[0]
        y = y + centre[1]
        for width in range(-draw_width // 2, draw_width // 2):
            x = centre[0] + width
            if 0 <= x < len(img_array) and 0 <= y < len(img_array[0]):
                if (x, y) != centre:
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


def auto_96well_detect(image_path: str|Path, minDist:int = None, minRadius:int = 30, maxRadius:int = 50, well_num:int = 96, output_path: str|Path = None):
    if minDist is None:
        minDist = minRadius * 2
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_circles_result.png"
    # 画像の読み込み
    image_path = Path(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = False
    left = 1
    right = 100
    well_df = pd.DataFrame()
    while right-left>0.1:
        param2 = (left+right)/2
        print(param2)
        # ハフ変換で円を検出
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist,
                                param1=50, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # 検出した円の座標を取得
        if circles is not None:
            circles = np.uint16(np.around(circles))
            well_centers = [(x, y, r) for x, y, r in circles[0]]

            # 検出した円を画像に描画
            output = image.copy()
            for (x, y, r) in circles[0]:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # 中心点

            # 結果を表示
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            # 検出したウェルの中心座標をデータフレーム化
            well_df = pd.DataFrame(well_centers, columns=["X", "Y", "R"])
            if len(well_df)<well_num:
                right = param2
                continue
            elif 96<len(well_df):
                left = param2
                continue
            else:
                detected=True
                well_df["A"] = 10000 - well_df["X"] - well_df["Y"]
                well_df["B"] = 10000 + well_df["X"] - well_df["Y"]
                well_df["C"] = 10000 + well_df["X"] + well_df["Y"]
                break

                # print(well_df)
                # tools.display_dataframe_to_user(name="Well Centers", dataframe=well_df)
        else:
            right = param2
            print("ウェルが検出できませんでした。パラメータを調整する必要があります。")
    print(f"{detected=}")
    print(well_df)
    plt.show()
    if detected:
        # Save the result
        cv2.imwrite(str(output_path), output)
        return well_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="96-well plate detection from an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("out_dir", type=str, help="Path to save the output file")
    parser.add_argument("--minRadius", type=int, default=90, help="Minimum radius of detected wells")
    parser.add_argument("--maxRadius", type=int, default=100, help="Maximum radius of detected wells")
    parser.add_argument("--minDist", type=int, default=None, help="Minimum distance between detected wells")
    args = parser.parse_args()
    img_path = args.image_path
    out_dir = Path(args.out_dir)
    minRadius = args.minRadius
    maxRadius = args.maxRadius
    if args.minDist is None:
        minDist = minRadius * 2

        
    img = cv2.imread(img_path)  # 画像の読み込み
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 色配置の変換 BGR→RGB
    img_array = np.asarray(img)  # numpyで扱える配列をつくる

    print(f"{img_array.shape=}")
    print(f"{minRadius=}, {maxRadius=}, {minDist=}")

    df = auto_96well_detect(image_path=img_path, minDist=minDist, minRadius=minRadius, maxRadius=maxRadius, output_path=out_dir/"circles_result.png")
    df.to_csv("well_centres.csv", index=False)

    A = df.loc[df["A"].idxmax(), ["Y", "X"]].tolist()
    B = df.loc[df["B"].idxmax(), ["Y", "X"]].tolist()
    C = df.loc[df["C"].idxmax(), ["Y", "X"]].tolist()
    print(f"{A=}, {B=}, {C=}")

    wells = calc_well_places(A, B, C, ROW_NUM, COL_NUM)
    colours = dict()
    r = 5
    for key, well in wells.items():
        # Get average colour of the well
        x, y = well
        well_img = img_array[x-r:x+r, y-r:y+r]
        avg_colour = np.median(well_img, axis=(0, 1))
        colours[key] = avg_colour
    for c in colours.values():
        print(c)
    df = pd.DataFrame(list(colours.items()), columns=["Well", "Colour"])
    df["R"] = df["Colour"].apply(lambda x: x[2])
    df["G"] = df["Colour"].apply(lambda x: x[1])
    df["B"] = df["Colour"].apply(lambda x: x[0])

    df.drop("Colour", axis=1, inplace=True)
    df.to_csv(out_dir/"well_colours.csv", index=False)


    for well in wells.values():
        img_array = create_cross(img_array, well, xy_len=40)

    plt.imshow(img_array)
    # plt.show()

    cv2.imwrite(str(out_dir / "well_plate.png"), img_array)
