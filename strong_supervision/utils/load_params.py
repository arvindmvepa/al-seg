import os
import yaml


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


if __name__ == "__main__":
    exp_params = load_yaml("exp_mac_strong.yml")
    print(exp_params)