import os
import shutil


def save_train_cfg(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    patient_dirs = os.listdir(data_dir)
    for patient_dir in patient_dirs:
        info_path = os.path.join(data_dir, patient_dir, "Info.cfg")
        save_info_path = os.path.join(save_dir, patient_dir + ".cfg")
        if os.path.exists(info_path):
            shutil.copyfile(info_path, save_info_path)

if __name__ == "__main__":
    data_dir = r"C:\Users\Arvind\Downloads\Resources"
    save_dir = r"C:\Users\Arvind\Documents\al-seg-arvind\wsl4mis_data\ACDC\ACDC_training_cfgs"
    save_train_cfg(data_dir, save_dir)
