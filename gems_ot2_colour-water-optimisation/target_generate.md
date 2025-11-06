```sh
scp -O opentron_main/ot_2_target.py  ot2beta:/data/user_storage
ssh ot2beta
```

```sh-ot2beta
opentrons_execute /data/user_storage/ot_2_target.py && exit 
```

```sh
sleep 3600 && /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/.venv/bin/python3  /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/main.py -o /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/TARGRT_DATA/TARGET_DATA1 -s 1 -r /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/targetratio.csv -t "Target" && /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/.venv/bin/python3  /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/main.py -o /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/TARGRT_DATA/TARGET_DATA2 -s 5 -r /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/targetratio.csv -t "Target"  && /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/.venv/bin/python3  /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/RGB_converter/main.py -o /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/TARGRT_DATA/TARGET_DATA3 -s 9 -r /Users/yuyaarai/Documents/Project/gems_ot2_colour-water-optimisation-simulator/targetratio.csv -t "Target"
```

```sh