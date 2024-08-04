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
    "ACDC_3D": {
        "label":
            {"data_root": os.path.join("wsl4mis_data",
                                       "ACDC"),
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
             "num_classes": 4,
             "train_results_dir": "",
             "train_logits_path": "train_preds.npz"
             }
    },
    "CHAOS_CT": {
        "scribble":
            {"data_root": os.path.join("wsl4mis_data",
                                       "CHAOS",
                                       "CT_LIVER"),
             "train_file": os.path.join("wsl4mis",
                                        "data",
                                        "CHAOS",
                                        "CT_LIVER",
                                        "train.txt"),
             "val_file": os.path.join("wsl4mis",
                                      "data",
                                      "CHAOS",
                                      "CT_LIVER",
                                      "val.txt"),
             "test_file": os.path.join("wsl4mis",
                                       "data",
                                       "CHAOS",
                                       "CT_LIVER",
                                       "test.txt"),
             "num_classes": 2,
             "train_results_dir": "",
             "train_logits_path": "train_preds.npz"
             },
        "label":
            {   "data_root": os.path.join("wsl4mis_data", 
                                          "CHAOS",
                                          "CT_LIVER"),
                "train_file": os.path.join("wsl4mis",
                                           "data",
                                           "CHAOS",
                                           "CT_LIVER",
                                           "train.txt"),
                "val_file": os.path.join("wsl4mis",
                                         "data",
                                         "CHAOS",
                                         "CT_LIVER",
                                         "val.txt"),
                "test_file": os.path.join("wsl4mis",
                                          "data",
                                          "CHAOS",
                                          "CT_LIVER",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    "LVSC": {
        "scribble":
            {"data_root": os.path.join("wsl4mis_data", "LVSC"),
             "train_file": os.path.join("wsl4mis",
                                        "data",
                                        "LVSC",
                                        "train.txt"),
             "val_file": os.path.join("wsl4mis",
                                      "data",
                                      "LVSC",
                                      "val.txt"),
             "test_file": os.path.join("wsl4mis",
                                       "data",
                                       "LVSC",
                                       "test.txt"),
             "num_classes": 2,
             "train_results_dir": "",
             "train_logits_path": "train_preds.npz"
             },
        "label":
            {   "data_root": os.path.join("wsl4mis_data", "LVSC"),
                "train_file": os.path.join("wsl4mis",
                                           "data",
                                           "LVSC",
                                           "train.txt"),
                "val_file": os.path.join("wsl4mis",
                                         "data",
                                         "LVSC",
                                         "val.txt"),
                "test_file": os.path.join("wsl4mis",
                                          "data",
                                          "LVSC",
                                          "test.txt"),
                "num_classes": 2,
                "train_results_dir": "",
                "train_logits_path": "train_preds.npz"
            },    
            },
    "MSCMR": {
        "scribble":
            {"data_root": os.path.join("wsl4mis_data", "MSCMR"),
             "train_file": os.path.join("wsl4mis",
                                        "data",
                                        "MSCMR",
                                        "train.txt"),
             "val_file": os.path.join("wsl4mis",
                                      "data",
                                      "MSCMR",
                                      "val.txt"),
             "test_file": os.path.join("wsl4mis",
                                       "data",
                                       "MSCMR",
                                       "test.txt"),
             "num_classes": 4,
             "train_results_dir": "",
             "train_logits_path": "train_preds.npz"
             },
    },
    "DAVIS": {
        "label":
            {"data_root": os.path.join("wsl4mis_data", "DAVIS"),
             "train_file": os.path.join("wsl4mis",
                                        "data",
                                        "DAVIS",
                                        "train.txt"),
             "val_file": os.path.join("wsl4mis",
                                      "data",
                                      "DAVIS",
                                      "val.txt"),
             "test_file": os.path.join("wsl4mis",
                                       "data",
                                       "DAVIS",
                                       "test.txt"),
             "num_classes": 2,
             "train_results_dir": "",
             "train_logits_path": "train_preds.npz"
             },
    },
    }