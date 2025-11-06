import re

def replace_values(template_path, output_path, color_num, total_volume, ratios, start_column):
    """
    Replace COLOR_NUM, TOTAL_VOLUME, and RATIO lines in the OT-2 template script.

    :param template_path: Path to the OT-2 template file.
    :param output_path: Path to save the modified OT-2 script.
    :param color_num: Number of colors to mix.
    :param total_volume: Total volume to distribute.
    :param ratios: List of ratios for each color.
    """
    # Read the template file
    with open(template_path, 'r') as file:
        content = file.read()

    # Replace COLOR_NUM
    content = re.sub(r"^COLOR_NUM\s*=\s*\d+", f"COLOR_NUM = {color_num}", content, flags=re.MULTILINE)

    # Replace TOTAL_VOLUME
    content = re.sub(r"^TOTAL_VOLUME\s*=\s*\d+", f"TOTAL_VOLUME = {total_volume}", content, flags=re.MULTILINE)

    # Replace RATIO
    ratio_string = ", ".join(map(str, ratios))
    content = re.sub(r"^RATIOS\s*=\s*\[.*\]", f"RATIOS = [{ratio_string}]", content, flags=re.MULTILINE)

    # Replace DUP_START
    content = re.sub(r"^DUP_START\s*=\s*\d+", f"DUP_START = {start_column}", content, flags=re.MULTILINE)

    # Replace TIP_SOURCE
    tip_source = (start_column - 1) // 4 + 7
    content = re.sub(r"^TIP_SOURCE\s*=\s*\d+", f"TIP_SOURCE = {tip_source}", content, flags=re.MULTILINE)

    # Write the modified content to the output file
    with open(output_path, 'w') as file:
        file.write(content)

    print(f"Generated OT-2 script saved to {output_path}")


# Example usage
if __name__ == "__main__":
    template_file = "ot_2_template.py"  # Path to the template file
    output_file = "ot_2_generated.py"  # Path for the generated file

    # Example input
    color_num = 3
    total_volume = 100
    ratios = [[1, 2, 1],[2, 1, 1],[1, 1, 2],[2, 2, 1],[1, 2, 2],[2, 1, 2],[1, 3, 1],[3, 1, 1],[1, 1, 3],[3, 2, 1],[2, 3, 1],[1, 3, 2]]

    replace_values(template_file, output_file, color_num, total_volume, ratios)
