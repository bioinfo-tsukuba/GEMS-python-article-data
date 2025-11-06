import string
from opentrons import protocol_api

# Metadata
metadata = {
    'protocolName': 'Color Mixing Based on Ratios with Duplicates - Tip Saving',
    'author': 'Assistant',
    'description': 'Mix color solutions based on given ratios into 96 well plate duplicates (A and B) for each column while minimizing tip usage.',
    'apiLevel': '2.14'  # Ensure this matches your OT-2 API version
}

COLOR_NUM = 3  # Number of colors to mix
TOTAL_VOLUME = 300  # Half of the mixing container volume (example: 400μl well capacity)
RATIOS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]  # RATIOS, 8 sets of ratios as default
DUP_NUM = 4  # Number of duplicates (A and B) for each column
DUP_START = 1  # Starting number for duplicates (A1, B1, A2, B2, ...)

# Protocol

def run(protocol: protocol_api.ProtocolContext):
    # Labware setup
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', 1)  # 96 well plate
    tube_rack = protocol.load_labware('opentrons_6_tuberack_falcon_50ml_conical', 2)  # 50ml tube rack
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', 3)  # Tip rack

    # Pipette setup
    pipette = protocol.load_instrument('P300_Single_GEN2', mount='left', tip_racks=[tiprack])

    # Define colors and ratios
    colors = [f'color_{i+1}' for i in range(COLOR_NUM)]
    total_volume = TOTAL_VOLUME


    # Define source tubes for each color (in tube rack positions)
    sources = {}
    for i in range(COLOR_NUM):
        if i < 3:
            # Assuming tube_rack has positions A1-A6 and B1-B6
            sources[f'color_{i+1}'] = tube_rack.wells_by_name()[f'A{i+1}']
        else:
            sources[f'color_{i+1}'] = tube_rack.wells_by_name()[f'B{i-2}']

    # Calculate volumes based on ratios for each set of ratios
    # Define destination wells (A1, B1, C1, D1, A2, B2, C2, D2, ...)
    color_to_destinations = {color: [] for color in colors}
    destination_wells = []
    volumes_list = []
    for idx, ratios in enumerate(RATIOS):
        total_ratio = sum(ratios)
        volumes = [total_volume * (ratio / total_ratio) for ratio in ratios]
        volumes_list.append(volumes)

        # Define destination wells for each set of ratios
        row = string.ascii_uppercase[idx % 8]  # Rows A to H
        destination_wells.append([])
        for col in range(DUP_NUM):
            col = col + DUP_START
            destination_wells[-1].append(plate.wells_by_name()[f'{row}{col}'])

    # Create a mapping from colors to list of (volume, well) tuples
    for idx, ratios in enumerate(RATIOS):
        volumes = volumes_list[idx]
        wells = destination_wells[idx]
        for well in wells:
            for color, volume in zip(colors, volumes):
                color_to_destinations[color].append((volume, well))

    # Shuffle the destination wells to minimize tip usage
    import random

    # Mixing process for each color
    for i, color in enumerate(colors):
        source = sources[color]
        destinations = color_to_destinations[color]
        random.shuffle(destinations)  # Shuffle the destinations to minimize tip usage
        
        if not destinations:
            continue  # Skip if there are no destinations for this color

        if i != 2:
            for volume, well in destinations:
                pipette.transfer(
                    volume,
                    source,
                    well,
                    new_tip='always',
                    blow_out=True,
                )
        else:
            for volume, well in destinations:
                pipette.transfer(
                    volume,
                    source,
                    well,
                    new_tip='always',
                    mix_after=(3, 80),
                    blow_out=True,
                )

