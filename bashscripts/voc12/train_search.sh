# search through hyper-params
file=bashscripts/voc12/train_spml_scribbles50_itags50.sh
GPUS=0
for params in "0.1 0.5" "0.2 0.75" "0.3 1.0" ; do
  IMG_SIM_LOSS_WEIGHT=$(echo $params | sed -E 's/(.*) (.*)/\1/')
  SEM_OCC_LOSS_WEIGHT=$(echo $params | sed -E 's/(.*) (.*)/\2/')
  export IMG_SIM_LOSS_WEIGHT
  export SEM_OCC_LOSS_WEIGHT
  source $file
done