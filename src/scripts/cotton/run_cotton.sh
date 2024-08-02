#!/bin/bash
#$ -j 
#$ -o /homes/ykohata/code/devml/homes/ypark/code/seg/trash
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new-ra

WORKDIR=/homes/ykohata/code/devml/homes/ypark/code/seg
echo "ok"

cd $WORKDIR/src

seed=51
python main.py voc \
    default.dataset_dir="/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/" \
    default.seed=$seed \
    learn.n_epoch=3 \
    augment.name=["ra"] \
    augment.ra.weight="single" \
    augment.ra.single="Cutout" \
    && python notify.py 1 || python notify.py 0