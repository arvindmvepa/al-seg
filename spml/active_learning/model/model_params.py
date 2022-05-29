import os


model_params = {
    "spml":
                 {
                     "box":
                      {
                          "train_split": "train+",
                          "left_base_im_list": "seambox_train+",
                          "right_base_im_list": "a6_th0.5_hed.txt",
                          "left_base_pim_list": "panoptic_train+",
                          "right_base_pim_list": "hed.txt",
                          "exec_script": "bashscripts/voc12/train_spml_box_al.sh",
                          "train_results_dir": os.path.join("stage1", "results", "train+", "semantic_gray")
                      },
                  "scribble":
                      {
                          "train_split": "train+",
                          "left_base_im_list": "scribble_train+",
                          "right_base_im_list": "d3_hed.txt",
                          "left_base_pim_list": "panoptic_train+",
                          "right_base_pim_list": "hed.txt",
                          "exec_script": "bashscripts/voc12/train_spml_scribble_al.sh",
                          "train_results_dir": os.path.join("stage1", "results", "train+", "semantic_gray")
                      },
                  },
    "a2gnn": {}
}