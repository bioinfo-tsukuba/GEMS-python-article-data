from pathlib import Path
import time
from config import config
from ssh_cmd import execute_remote_script, execute_remote_script_interactive_mode, execute_remote_transfer, execute_remote_download
from datetime import datetime

if __name__ == "__main__":
    # Configuration
    HOST = config.host
    PORT = config.port
    USERNAME = config.username
    KEY_PATH = config.key_path
    OT2_CODE_FILE = config.ot2_code_file
    OT2_CALIBRATION_FILE = config.ot2_calibration_file
    DESTINATION_DIR = config.destination_dir
    KEY_TYPE = config.key_type
    DEBUG = config.debug
    PASSWORD = config.password

    # get local time
    date_time = datetime.now().astimezone()
    date_time_str = date_time.strftime(f'%Y_%m_%dT%H_%M_%S_TZ_{date_time.tzinfo}')
    local_files = [OT2_CODE_FILE]
    remote_dir = DESTINATION_DIR
    remote_file_path = str(Path(DESTINATION_DIR) / Path(OT2_CODE_FILE).name)

    local_dir = Path(__file__).parent
    remote_output_file = f"/data/user_storage/ot2_output_{date_time_str}.txt"
    if DEBUG:
        execute_command = f"opentrons_simulate {remote_file_path} | tee -a  {remote_output_file}"
    else:
        execute_command = f"opentrons_execute {remote_file_path} | tee -a  {remote_output_file}"
    print(f"{execute_command=}")
    time.sleep(100)



    execute_remote_script_interactive_mode(
        host=HOST,
        port=PORT,
        username=USERNAME,
        key_path=KEY_PATH,
        command="ls",
        key_type=KEY_TYPE,
        password=PASSWORD
    )


    if OT2_CALIBRATION_FILE is not None:
        execute_remote_transfer(
            host=HOST,
            port=PORT,
            username=USERNAME,
            key_path=KEY_PATH,
            local_files=[OT2_CALIBRATION_FILE],
            remote_dir="/data",
            key_type=KEY_TYPE,
            password=PASSWORD
        )



    execute_remote_transfer(
        host=HOST,
        port=PORT,
        username=USERNAME,
        key_path=KEY_PATH,
        local_files=local_files,
        remote_dir=remote_dir,
        key_type=KEY_TYPE,
        password=PASSWORD
    )

    execute_remote_script_interactive_mode(
        host=HOST,
        port=PORT,
        username=USERNAME,
        key_path=KEY_PATH,
        command=execute_command,
        key_type=KEY_TYPE,
        password=PASSWORD
    )



    execute_remote_download(
        host=HOST,
        port=PORT,
        username=USERNAME,
        key_path=KEY_PATH,
        remote_files=[remote_output_file],
        local_dir=local_dir,
        key_type=KEY_TYPE,
        password=PASSWORD
    )
