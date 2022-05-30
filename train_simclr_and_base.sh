file=bashscripts/voc12/train_spml_scribble.sh
for PRETRAINED in "./snapshots/imagenet/trained/simclr_resnet101_pretrained1.pth" "./snapshots/imagenet/trained/resnet-101-cuhk.pth"; do
  export PRETRAINED
  source $file
done

file2=bashscripts/voc12/train_contrastive_learning.sh
source $file2
