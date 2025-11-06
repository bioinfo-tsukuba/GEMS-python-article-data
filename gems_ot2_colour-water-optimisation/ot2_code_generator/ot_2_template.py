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
TOTAL_VOLUME = 200  # Half of the mixing container volume (example: 400ul well capacity)
RATIOS = [[1, 2, 1],[2, 1, 1],[1, 1, 2],[2, 2, 1],[1, 2, 2],[2, 1, 2],[1, 3, 1],[3, 1, 1]]  # RATIOS, 8 sets of ratios as default
DUP_NUM = 4  # Number of duplicates (A and B) for each column
DUP_START = 9  # Starting number for duplicates (A1, B1, A2, B2, ...)
TIP_SOURCE = 7  # Tip rack position

# Protocol

def run(protocol: protocol_api.ProtocolContext):
    # Labware setup
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', 1)  # 96 well plate
    tube_rack = protocol.load_labware('opentrons_6_tuberack_falcon_50ml_conical', 2)  # 50ml tube rack
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', TIP_SOURCE)  # 300ul tip rack

    # Pipette setup
    pipette = protocol.load_instrument('P300_Single_GEN2', mount='right', tip_racks=[tiprack])
    pipette.flow_rate.aspirate = 100
    pipette.flow_rate.dispense = 100

    # Define colors and ratios
    colors = [f'color_{i+1}' for i in range(COLOR_NUM)]
    total_volume = TOTAL_VOLUME


    # Define source tubes for each color (in tube rack positions)
    sources = {}
    for i in range(COLOR_NUM):
        if i < 3:
            # Assuming tube_rack has positions A1-A6 and B1-B6
            sources[colors[i]] = tube_rack.wells_by_name()[f'A{i+1}']
        else:
            sources[colors[i]] = tube_rack.wells_by_name()[f'B{i-2}']

    # Define a list of (volume, destination_well, source_well) tuples for each color
    volumes = [[] for _ in range(COLOR_NUM)]
    for color_i, color in enumerate(colors):
        for ratio in RATIOS:
            volumes[color_i].append(total_volume * ratio[color_i] / sum(ratio))
    # display the volumes
    print(volumes)

    operation_list = []
    for volume, color in zip(volumes, colors):
        for row_i, vol in enumerate(volume):
            row = string.ascii_uppercase[row_i]
            for dup_i in range(DUP_NUM):
                col = DUP_START + dup_i
                well = f'{row}{col}'
                operation_list.append((vol, plate.wells_by_name()[well], sources[color]))


    # Sort the operation list by volume to minimize transfer failures
    operation_list.sort(key=lambda x: x[0], reverse=True)

    well_volumes = dict()
    # Perform the transfers
    for volume, destination, source in operation_list:
        if volume <= 1:
            continue
        well_volume = well_volumes.get(destination, 0)
        if well_volume > 0:
            pipette.transfer(
                volume,
                source,
                destination,
                new_tip='always',
                mix_after=(3, well_volume*0.9),
            )
        else:
            pipette.transfer(
                volume,
                source,
                destination,
                new_tip='always',
            )
        well_volumes[destination] = well_volume + volume