#!/bin/bash
# search through hyper-params
file=bashscripts/voc12/train_spml_box_al.sh
for params in "1500 1500 0.05 0" "1500 1500 0.05 1" "1500 1500 0.05 2" "1500 1500 0.05 3" "1500 1500 0.05 4" ; do
  SNAPSHOT_STEP=$(echo $params | sed -E 's/(.*) (.*) (.*) (.*)/\1/')
  MAX_ITERATION=$(echo $params | sed -E 's/(.*) (.*) (.*) (.*)/\2/')
  AL_PROP=$(echo $params | sed -E 's/(.*) (.*) (.*) (.*)/\3/')
  MODEL_NO=$(echo $params | sed -E 's/(.*) (.*) (.*) (.*)/\4/')

  export AL_PROP
  export MODEL_NO
  source $file > "box_AL_PROP${AL_PROP}_MODEL_NO${MODEL_NO}.results" 2> "box_AL_PROP${AL_PROP}_MODEL_NO${MODEL_NO}.error"
done
