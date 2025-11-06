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
TOTAL_VOLUME = 200  # Half of the mixing container volume (example: 400μl well capacity)
RATIOS = [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5],
          [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5],
          [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]]  # RATIOS, 8 sets of ratios as default
DUP_NUM = 4  # Number of duplicates (A and B) for each column
DUP_START = 1  # Starting number for duplicates (A1, B1, A2, B2, ...)

# Protocol

def run(protocol: protocol_api.ProtocolContext):
    # Labware setup
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', 1)  # 96 well plate
    tube_rack = protocol.load_labware('opentrons_6_tuberack_falcon_50ml_conical', 2)  # 50ml tube rack
    tiprack1 = protocol.load_labware('opentrons_96_tiprack_300ul', 7)  # Tip rack
    tiprack2 = protocol.load_labware('opentrons_96_tiprack_300ul', 8)  # Tip rack
    tiprack3 = protocol.load_labware('opentrons_96_tiprack_300ul', 9)  # Tip rack
    tipracks = [tiprack1, tiprack2, tiprack3]  # Combine tip racks

    # Pipette setup
    pipette = protocol.load_instrument('P300_Single_GEN2', mount='right', tip_racks=tipracks)
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
    # dupstart = DUP_START
    for volume, color in zip(volumes, colors):
        dupstart = DUP_START
        for row_i, vol in enumerate(volume):
            row = string.ascii_uppercase[row_i%8]  # Rows A to H
            if row_i != 0 and row_i % 8 == 0:
                dupstart += DUP_NUM
            for dup_i in range(DUP_NUM):
                col = dupstart + dup_i
                well = f'{row}{col}'
                operation_list.append((vol, plate.wells_by_name()[well], sources[color]))


    # Sort the operation list by volume to minimize transfer failures
    operation_list.sort(key=lambda x: x[0], reverse=True)

    well_volumes = dict()
    # Perform the transfers
    for volume, destination, source in operation_list:
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
    
    # 最後にコマンド一覧をダンプ
    for line in protocol.commands():
        print(line)
