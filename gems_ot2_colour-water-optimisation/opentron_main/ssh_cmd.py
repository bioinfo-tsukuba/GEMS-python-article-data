from pathlib import Path
import re
import tempfile
import paramiko
import os
import sys
from config import config
import scp

def execute_remote_script(host, port, username, key_path, command: str, key_type: str, password = None):
    """
    Connects to a remote host via SSH, executes a shell script, and exits.

    :param host: Remote host IP or hostname.
    :param port: SSH port number (default is 22).
    :param username: SSH username.
    :param key_path: Path to the SSH private key.
    :param script_path: Path to the shell script on the remote host.
    :param script_args: List of arguments to pass to the shell script.
    """
    # Initialize the SSH client
    ssh = paramiko.SSHClient()
    
    # Automatically add the remote server's SSH key (not recommended for production)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Load the private key
        if key_type == "ed25519":
            key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "rsa":
            key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "ecdsa":
            key = paramiko.ECDSAKey.from_private_key_file(os.path.expanduser(key_path))
        else:
            raise ValueError("Invalid key file format.")
        
        # Connect to the remote host
        print(f"Connecting to {username}@{host}:{port}")
        ssh.connect(hostname=host, port=port, username=username, pkey=key, password = password)
        print("Connection established.")

        print(f"Executing script: {command}")
        
        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(command)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_out, \
            tempfile.NamedTemporaryFile(delete=False) as tmp_err:
            tmp_out.write(stdout.read())   # バイナリをそのまま保存
            tmp_err.write(stderr.read())
            tmp_out_path = tmp_out.name
            tmp_err_path = tmp_err.name

        exit_status = stdout.channel.recv_exit_status()

        # あとから一括で読み込んでデコード
        with open(tmp_out_path, 'rb') as f_out:
            binary_data = f_out.read()
            try:
                decoded_out = binary_data.decode('utf-8')
            except UnicodeDecodeError:
                decoded_out = binary_data.decode('utf-8', errors='ignore')

        with open(tmp_err_path, 'rb') as f_err:
            binary_err = f_err.read()
            try:
                decoded_err = binary_err.decode('utf-8')
            except UnicodeDecodeError:
                decoded_err = binary_err.decode('utf-8', errors='ignore')

        stdout_str = decoded_out
        stderr_str = decoded_err

        # Wait for the command to complete
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            print("Script executed successfully.")
            print("Output:")
            print(stdout_str)
        else:
            print(f"Script failed with exit status {exit_status}.")
            print("Error Output:")
            print(stderr_str)
        
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        sys.exit(1)
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        sys.exit(1)
    except Exception as e:
        print(f"Operation error: {e}")
        sys.exit(1)
    finally:
        # Close the SSH connection
        ssh.close()
        print("SSH connection closed.")

def execute_remote_script_interactive_mode(host, port, username, key_path, command: str, key_type: str, password = None):
    """
    Connects to a remote host via SSH, executes a shell script, and exits.

    :param host: Remote host IP or hostname.
    :param port: SSH port number (default is 22).
    :param username: SSH username.
    :param key_path: Path to the SSH private key.
    :param script_path: Path to the shell script on the remote host.
    :param script_args: List of arguments to pass to the shell script.
    """
    # Initialize the SSH client
    ssh = paramiko.SSHClient()
    
    # Automatically add the remote server's SSH key (not recommended for production)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Load the private key
        if key_type == "ed25519":
            key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "rsa":
            key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "ecdsa":
            key = paramiko.ECDSAKey.from_private_key_file(os.path.expanduser(key_path))
        else:
            raise ValueError("Invalid key file format.")
        
        # Connect to the remote host
        print(f"Connecting to {username}@{host}:{port}")
        ssh.connect(hostname=host, port=port, username=username, pkey=key, password = password)
        chan = ssh.invoke_shell()
        print("Connection established.")

        print(f"Executing script: {command}")


        # 一時ファイルにバイナリログを保存するハンドルを作成
        tmp_buf = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp_buf.name

        # プロンプトが返るまでバイナリとして受信し続ける
        buff = b''
        while True:
            recv = chan.recv(4096)
            if not recv:
                break
            tmp_buf.write(recv)
            buff += recv
            if buff.endswith(b'# '):
                break

        print("Prompt received. Sending command:")
        print(command)
        chan.send((command + "\n").encode())

        # コマンド実行後の出力が返るまで同じくバイナリ受信
        buff = b''
        while True:
            recv = chan.recv(4096)
            if not recv:
                break
            tmp_buf.write(recv)
            buff += recv
            # 再びプロンプトが返ってきたらコマンド完了とみなす
            if buff.endswith(b'# '):
                break

        tmp_buf.close()  # ファイルをクローズして書き込み完了

        # 一時ファイルからまとめてデコード
        with open(tmp_path, 'rb') as f:
            all_data = f.read()
            try:
                decoded_all = all_data.decode('utf-8')
            except UnicodeDecodeError:
                decoded_all = all_data.decode('utf-8', errors='ignore')

        print("Script executed successfully. Full log:")
        print(decoded_all)
        
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        sys.exit(1)
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        sys.exit(1)
    except Exception as e:
        print(f"Operation error: {e}")
        sys.exit(1)
    finally:
        # Close the SSH connection
        ssh.close()
        print("SSH connection closed.")



def execute_remote_transfer(host, port, username, key_path, local_files, remote_dir, key_type: str, password = None):
    """
    Connects to a remote host via SSH, executes a shell script, and exits.

    :param host: Remote host IP or hostname.
    :param port: SSH port number (default is 22).
    :param username: SSH username.
    :param key_path: Path to the SSH private key.
    :param script_path: Path to the shell script on the remote host.
    :param script_args: List of arguments to pass to the shell script.
    """
    # Initialize the SSH client
    ssh = paramiko.SSHClient()
    
    # Automatically add the remote server's SSH key (not recommended for production)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        local_files = [Path(f) for f in local_files]
        remote_dir = Path(remote_dir)
        # Load the private key
        if key_type == "ed25519":
            key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "rsa":
            key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "ecdsa":
            key = paramiko.ECDSAKey.from_private_key_file(os.path.expanduser(key_path))
        else:
            raise ValueError("Invalid key file format.")
        
        # Connect to the remote host
        print(f"Connecting to {username}@{host}:{port}")
        ssh.connect(hostname=host, port=port, username=username, pkey=key, password = password)
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            scp_client.put(
                files=local_files,
                remote_path=remote_dir,
            )

        print(f"Files transferred successfully.")
        print(f"Files {local_files} transferred to {username}@{host}:{remote_dir}")
        
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        sys.exit(1)
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        sys.exit(1)
    except Exception as e:
        print(f"Operation error: {e}")
        sys.exit(1)
    finally:
        # Close the SSH connection
        ssh.close()
        print("SSH connection closed.")


def execute_remote_download(host, port, username, key_path, remote_files, local_dir, key_type: str, password = None):
    """
    Connects to a remote host via SSH, executes a shell script, and exits.

    :param host: Remote host IP or hostname.
    :param port: SSH port number (default is 22).
    :param username: SSH username.
    :param key_path: Path to the SSH private key.
    :param script_path: Path to the shell script on the remote host.
    :param script_args: List of arguments to pass to the shell script.
    """
    # Initialize the SSH client
    ssh = paramiko.SSHClient()
    
    # Automatically add the remote server's SSH key (not recommended for production)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        remote_files = [Path(f) for f in remote_files]
        local_dir = Path(local_dir)
        # Load the private key
        if key_type == "ed25519":
            key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "rsa":
            key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        elif key_type == "ecdsa":
            key = paramiko.ECDSAKey.from_private_key_file(os.path.expanduser(key_path))
        else:
            raise ValueError("Invalid key file format.")
        
        # Connect to the remote host
        print(f"Connecting to {username}@{host}:{port}")
        ssh.connect(hostname=host, port=port, username=username, pkey=key, password = password)
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            for remote_file in remote_files:
                scp_client.get(
                    remote_path=remote_file,
                    local_path=local_dir/remote_file.name,
                )

        print(f"Files downloaded successfully.")
        print(f"Files {remote_files} downloaded to {local_dir}")
        
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        sys.exit(1)
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        sys.exit(1)
    except Exception as e:
        print(f"Operation error: {e}")
        sys.exit(1)
    finally:
        # Close the SSH connection
        ssh.close()
        print("SSH connection closed.")