import subprocess
import os

def load_virtualenv(virtualenv_path):
    output = subprocess.check_output(['bash', '-c', f'source {virtualenv_path} && env'])
    env_vars = dict(line.split('=', 1) for line in output.decode().splitlines())
    os.environ = env_vars