#!/bin/bash
# search through hyper-params
file=bashscripts/voc12/train_spml_scribble.sh
for params in "6 0.2" "6 0.5" "12 0.2" "12 0.4" ; do
  WORD_SIM_CONCENTRATION=$(echo $params | sed -E 's/(.*) (.*)/\1/')
  WORD_SIM_LOSS_WEIGHT=$(echo $params | sed -E 's/(.*) (.*)/\2/')
  export WORD_SIM_CONCENTRATION
  export WORD_SIM_LOSS_WEIGHT
  source $file > "scribble_wordsim_${WORD_SIM_CONCENTRATION}_${WORD_SIM_LOSS_WEIGHT}.results" 2> "scribbles_wordsim_${WORD_SIM_CONCENTRATION}_${WORD_SIM_LOSS_WEIGHT}.error"
done
