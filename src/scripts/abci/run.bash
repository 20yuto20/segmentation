#!/bin/bash
#$ -l rt_G.small=1
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

mkdir -p $SGE_LOCALDIR/datasets
cp -v "/groups/gaa50073/kohata-yuto/voc_datasets.tar.gz" $SGE_LOCALDIR/datasets/
cd $SGE_LOCALDIR/datasets
tar -I pigz -xf "voc_datasets.tar.gz"
ls

cd $WORKDIR

# case $SGE_TASK_ID in
#     1) seed=502;;
#     2) seed=503;;
#     *) echo "Invalid task ID"; exit 1;;
# esac

seed=1

python main.py voc_abci \
    default.device_id=0 \
    default.dataset_dir="$SGE_LOCALDIR/datasets/" \
    default.seed=$seed \
    augment.name=["ra"] \
    augment.ra.weight="single" \
    augment.ra.single="ShearX"

# 実行するとき
# $SGE_TASK_IDを設定するとき，
# qsub -g gaa50073 -t 1-2:1 /groups/gaa50073/park-yuna/cont/src/scripts/s0725/aff_inv.sh
# $SGE_TASK_IDを設定しないとき，
# qsub -g gaa50073 /groups/gaa50073/park-yuna/cont/src/scripts/s0725/aff_inv.sh