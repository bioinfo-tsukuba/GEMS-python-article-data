import argparse
import pandas as pd
from gen import replace_values
import subprocess
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='This is sample argparse script')
    parser.add_argument('-t', '--template', help='Template file path', required=True)
    parser.add_argument('-o', '--output', help='Output file path', required=True)
    parser.add_argument('-r', '--ratiospath', help='Ratios file path', required=True)
    parser.add_argument('-s', '--start', help='Start column number', type=int, required=True)
    parser.add_argument('-c', '--check', help='Check if the output file is correct', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    ratios = pd.read_csv(args.ratiospath)
    ratios = ratios.values.tolist()

    replace_values(template_path=args.template, output_path=args.output, color_num=len(ratios[0]), total_volume=200, ratios=ratios, start_column=args.start)

    # ```shell
    # opentrons_simulate -f {output_file_path}
    # ```

    # Run
    if args.check:
        try:
            output_dir = Path(args.output).parent
            output_str = subprocess.run(['opentrons_simulate', args.output], capture_output=True, text=True)
            # Output the result as text file
            with open(output_dir / 'check_output.txt', 'w') as f:
                f.write(output_str.stdout)
        except Exception as e:
            print(e)
