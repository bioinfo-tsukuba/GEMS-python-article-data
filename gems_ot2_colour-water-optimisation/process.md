# Set .env file

```./opentrons_main/.env
HOST=
PORT=
USERNAME=
KEY_PATH=
OT2_CODE_FILE=
DESTINATION_DIR=
KEY_TYPE=
DEBUG=
```

```zsh
pyenv local 3.12.7 && python -m venv .venv && source .venv/bin/activate && pip install pip git+https://github.com/CAB314/GEMS-python.git#main && deactivate
```

```zsh
mkdir -p opentron_main
cd opentron_main && pyenv local 3.12.7 && python -m venv .venv && source .venv/bin/activate && pip install pydantic_settings && deactivate && cd ..
```

```zsh
mkdir -p ot2_code_generator
cd ot2_code_generator && pyenv local 3.12.7 && python -m venv .venv && source .venv/bin/activate && deactivate && cd ..
```

```zsh
mkdir -p RGB_converter
cd RGB_converter && pyenv local 3.12.7 && python -m venv .venv && source .venv/bin/activate && deactivate && cd ..
```



```zsh
mkdir -p absorbance_processor
cd absorbance_processor && pyenv local 3.12.7 && python -m venv .venv && source .venv/bin/activate && pip install pandas && deactivate && cd ..
```

```zsh
cd opentron_main &&  source .venv/bin/activate && pip freeze > requirements.txt && deactivate && cd ..
```

```zsh
cd ot2_code_generator &&  source .venv/bin/activate && pip freeze > requirements.txt && deactivate && cd ..
```

```ot2_experiment/mode/mode.txt
add_machines
```


```ot2_experiment/mode/mode_add_machines.txt
0,OT2
1,Human

```

```ot2_experiment/mode/mode.txt
add_experiments
```

```ot2_experiment/mode/mode_add_experiments.txt
ot2_setting.gen_ot2_cwo_experiment

```

```ot2_experiment/mode/mode.txt
loop
```

