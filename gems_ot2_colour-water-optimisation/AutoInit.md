exec $SHELL -l

touch ot2_experiment/mode/mode.txt && echo "add_machines" > ot2_experiment/mode/mode.txt \
&& touch ot2_experiment/mode/mode_add_machines.txt && echo "0,OT2" > ot2_experiment/mode/mode_add_machines.txt && echo "1,Human" >> ot2_experiment/mode/mode_add_machines.txt

sleep  10

touch ot2_experiment/mode/mode.txt && echo "add_experiments" > ot2_experiment/mode/mode.txt \
&& touch ot2_experiment/mode/mode_add_experiments.txt && echo "ot2_setting.gen_ot2_cwo_experiment" > ot2_experiment/mode/mode_add_experiments.txt

sleep 10

touch ot2_experiment/mode/mode.txt && echo "loop" > ot2_experiment/mode/mode.txt

./.venv/bin/python3 ./auto_process.py

touch ot2_experiment/mode/mode.txt && echo "eof" > ot2_experiment/mode/mode.txt