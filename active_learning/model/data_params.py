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
    "CHAOS": {
        "label":
            {   "data_root": os.path.join("chaos_data"), 
                "train_file": os.path.join("chaos",
                                           "data",
                                           "train.txt"),
                "val_file": os.path.join("chaos",
                                         "data",
                                         "val.txt"),
                "test_file": os.path.join("chaos",
                                          "data",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    "LVSC": {
        "label":
            {   "data_root": os.path.join("LVSC_data"), 
                "train_file": os.path.join("LVSC",
                                           "data",
                                           "train.txt"),
                "val_file": os.path.join("LVSC",
                                         "data",
                                         "val.txt"),
                "test_file": os.path.join("LVSC",
                                          "data",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    }