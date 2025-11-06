import argparse
from pathlib import Path
import pandas as pd
import string

def get_args():
    parser = argparse.ArgumentParser(description='This is sample argparse script')
    parser.add_argument('-c', '--colour', help='Colour file path', required=True)
    parser.add_argument('-r', '--ratiospath', help='Ratios file path', required=True)
    parser.add_argument('-o', '--output', help='Output file directory', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    colour_df = pd.read_csv(args.colour)
    colour_df["column"] = colour_df["well"].apply(lambda x: x[1:])
    colour_df["row"] = colour_df["well"].apply(lambda x: x[0])
    #   well     R      G     B
    #   A5  75.5  102.5  59.5
    ratios = pd.read_csv(args.ratiospath)
    ratios["row"] = "A"
    for i, ratio_set in enumerate(ratios.values):
        row_alphabet = string.ascii_uppercase[i]
        ratios.loc[i, "row"] = row_alphabet
    #        Color1_ratio  Color2_ratio  Color3_ratio
    #        0.140715      0.793967      0.065319

    merged_df = pd.merge(colour_df, ratios, on="row")

    print(merged_df)

    merged_df.to_csv(Path(args.output)/"merged.csv", index=False)



