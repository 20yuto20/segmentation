#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o /groups/gaa50073/kohata-yuto/segmentation/src/trash/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.9
source ~/.bashrc
conda activate seg-env

WORKDIR="/groups/gaa50073/kohata-yuto/segmentation/src/"
echo "ok"
echo $WORKDIR

mkdir -p $SGE_LOCALDIR/dataset
cp -v "/groups/gaa50073/kohata-yuto/voc_datasets.tar.gz" $SGE_LOCALDIR/dataset/
cd $SGE_LOCALDIR/dataset
tar -I pigz -xf "voc_datasets.tar.gz"
echo "unzip the dataset"

cd $WORKDIR

seed=1

python main.py voc \
    default.device_id=0 \
    default.dataset_dir="$SGE_LOCALDIR/dataset/voc_aug/" \
    learn.n_epoch=1 \
    learn.batch_size=16 \
    default.seed=$seed \
    augment.name=["nan"] \
    augment.ra.weight="single" \
    augment.ra.single="ShearX" \
    && python notify.py 0 || python notify.py 1