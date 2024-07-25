#!/bin/bash
#$ -j 
#$ -o /homes/ykohata/code/devml/homes/ypark/code/seg/trash/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new-ra

WORKDIR="/homes/ykohata/code/devml/homes/ypark/code/seg/src_cls/src"
echo "ok"

cd $WORKDIR



seed=111
python main.py  voc \
    default.seed=$seed \
    augment.name=["hflip","rcrop","cutout"] \
    learn.n_epoch=3 \
    augment.hp.cutout_p=1.0 \
    augment.hp.rcrop_pad=14

