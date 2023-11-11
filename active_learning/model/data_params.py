import os 


data_params = {
    "ACDC": {
        "scribble":
                      {   "data_root": os.path.join("wsl4mis_data",
                                                    "ACDC"),
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
                          "num_classes": 4,
                          "train_results_dir": "",
                          "train_logits_path": "train_preds.npz"
                      },
        "label": 
            {             "data_root": os.path.join("wsl4mis_data",
                                                    "ACDC"),
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
                          "num_classes": 4,
                          "train_results_dir": "",
                          "train_logits_path": "train_preds.npz"
                      } 
            },
    "CHAOS_CT": {
        "label":
            {   "data_root": os.path.join("wsl4mis_data", "CHAOS"),
                "train_file": os.path.join("CHAOS",
                                           "CT_LIVER",
                                           "train.txt"),
                "val_file": os.path.join("chaos",
                                         "CT_LIVER",
                                         "val.txt"),
                "test_file": os.path.join("chaos",
                                          "CT_LIVER",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    "LVSC": {
        "label":
            {   "data_root": os.path.join("wsl4mis_data", "LVSC"),
                "train_file": os.path.join("LVSC",
                                           "train.txt"),
                "val_file": os.path.join("LVSC",
                                         "val.txt"),
                "test_file": os.path.join("LVSC",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    }