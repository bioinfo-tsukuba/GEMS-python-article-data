```sh
scp -O opentron_main/ot_2_target.py  ot2beta:/data/user_storage
ssh ot2beta
```

```sh-ot2beta
tmux
opentrons_execute /data/user_storage/ot_2_target.py && exit
```

```sh
./RGB_converter/.venv/bin/python3 ./RGB_converter/take_movie.py --duration 12000 --interval 20 --output RGB_converter/sample_movie.mp4 &&  ./RGB_converter/main.py -o ./TARGRT_DATA/TARGET_DATA1 -s 1 -r ./targetratio.csv -t "Target" && ./RGB_converter/.venv/bin/python3  ./RGB_converter/main.py -o ./TARGRT_DATA/TARGET_DATA2 -s 5 -r ./targetratio.csv -t "Target"  && ./RGB_converter/.venv/bin/python3  ./RGB_converter/main.py -o ./TARGRT_DATA/TARGET_DATA3 -s 9 -r ./targetratio.csv -t "Target" && 
git add .
git commit -m "Auto commit of real-world-experiment at $(date '+%Y-%m-%d %H:%M:%S')"
git push
```
