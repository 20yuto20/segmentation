#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=02:00:00
#$ -j y
#$ -o /groups/gaa50073/kohata-yuto/segmentation/src_cls/src/trash/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.9
source ~/.bashrc
conda activate seg-env

WORKDIR="/groups/gaa50073/kohata-yuto/segmentation/src_cls/src/"
echo "ok"
echo $WORKDIR

mkdir -p $SGE_LOCALDIR/dataset
cp -v "/groups/gaa50073/kohata-yuto/voc_datasets.tar.gz" $SGE_LOCALDIR/dataset/
cd $SGE_LOCALDIR/dataset
tar -I pigz -xf "voc_datasets.tar.gz"
echo "unzip the dataset"

cd $WORKDIR

case $SGE_TASK_ID in
    1) seed=2024;;
    2) seed=2025;;
    3) seed=2026;;
    *) echo "Invalid task ID"; exit 1;;
esac

# seed=301

python main.py voc \
    default.device_id=0 \
    default.home_dir="/groups/gaa50073/kohata-yuto/segmentation/" \
    default.dataset_dir="$SGE_LOCALDIR/dataset/voc/VOCdevkit" \
    learn.n_epoch=50 \
    learn.batch_size=32 \
    default.seed=$seed \
    augment.name=["ra"] \
    augment.ra.weight="random" \
    augment.ra.single="Invert" \
    && python notify.py 0 || python notify.py 1

# 実行するとき
# $SGE_TASK_IDを設定するとき，
# qsub -g gaa50073 -t 1-2:1 /groups/gaa50073/park-yuna/cont/src/scripts/s0725/aff_inv.sh
# $SGE_TASK_IDを設定しないとき，
# qsub -g gaa50073 /groups/gaa50073/park-yuna/cont/src/scripts/s0725/aff_inv.sh