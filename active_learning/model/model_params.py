import os


model_params = {
    "spml":
                 {
                     "box":
                      {
                          "train_split": "train+",
                          "left_base_im_list": "seambox_train+",
                          "right_base_im_list": "a6_th0.5_hed.txt",
                          "orig_train_im_list_file": os.path.join("spml", "datasets", "voc12", "seambox_train+_a6_th0.5_hed.txt"),
                          "left_base_pim_list": "panoptic_train+",
                          "right_base_pim_list": "hed.txt",
                          "orig_train_pim_list_file" : os.path.join("spml", "datasets", "voc12",  "panoptic_train+_hed.txt"),
                          "val_pim_list": os.path.join("spml", "datasets", "voc12", "panoptic_val.txt"),
                          "exec_script": "spml/bashscripts/voc12/train_spml_box_al.sh",
                          "train_results_dir": os.path.join("stage1", "results", "train+", "semantic_gray"),
                          "train_logits_path": os.path.join("stage1", "results", "train+", "logits", "save_preds.npz")
                      },
                  "scribble":
                      {
                          "train_split": "train+",
                          "left_base_im_list": "scribble_train+",
                          "right_base_im_list": "d3_hed.txt",
                          "orig_train_im_list_file": os.path.join("spml", "datasets", "voc12", "scribble_train+_d3_hed.txt"),
                          "left_base_pim_list": "panoptic_train+",
                          "right_base_pim_list": "hed.txt",
                          "orig_train_pim_list_file" : os.path.join("spml", "datasets", "voc12",  "panoptic_train+_hed.txt"),
                          "val_pim_list": os.path.join("spml", "datasets", "voc12", "panoptic_val.txt"),
                          "exec_script": "spml/bashscripts/voc12/train_spml_scribble_al.sh",
                          "train_results_dir": os.path.join("stage1", "results", "train+", "semantic_gray"),
                          "train_logits_path": os.path.join("stage1", "results", "train+", "logits", "save_preds.npz")
                      },
                  },
    "dpmls":                  {
                  "scribble":
                      {
                          "train_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "train.txt"),
                          "val_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "val.txt"),
                          "test_file": os.path.join("wsl4mis",
                                                   "data",
                                                   "ACDC",
                                                   "test.txt"),
                          "train_results_dir": "",
                          "train_logits_path": "train_preds.npz"
                      },
                  },
    "strong":            {
                  "label":
                      {
                          "train_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "train.txt"),
                          "val_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "val.txt"),
                          "test_file": os.path.join("wsl4mis",
                                                   "data",
                                                   "ACDC",
                                                   "test.txt"),
                          "train_results_dir": "",
                          "train_logits_path": "train_preds.npz"
                      },
                  },
    "strong_3d": {
        "label":
            {
                "train_file": os.path.join("wsl4mis",
                                           "data",
                                           "ACDC",
                                           "train_3d.txt"),
                "val_file": os.path.join("wsl4mis",
                                         "data",
                                         "ACDC",
                                         "val.txt"),
                "test_file": os.path.join("wsl4mis",
                                          "data",
                                          "ACDC",
                                          "test.txt"),
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },
    },
    "db_dpmls":            {
                  "scribble":
                      {
                          "train_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "train.txt"),
                          "val_file": os.path.join("wsl4mis",
                                                    "data",
                                                    "ACDC",
                                                    "val.txt"),
                          "test_file": os.path.join("wsl4mis",
                                                   "data",
                                                   "ACDC",
                                                   "test.txt"),
                          "train_results_dir": "",
                          "train_logits_path": "train_preds.npz"
                      },
                  },

    "a2gnn": {}
}