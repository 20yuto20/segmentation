#!/bin/bash
#$ -j 
#$ -o /homes/ykohata/code/devml/homes/ypark/code/seg/src/trash/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new-ra

WORKDIR=/homes/ykohata/code/devml/homes/ypark/code/seg
echo "ok"

cd $WORKDIR/src

seed=2026
python main.py voc \
    default.device_id=0 \
    default.dataset_dir="/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/" \
    learn.n_epoch=50 \
    learn.batch_size=8 \
    default.seed=$seed \
    default.num_workers=10 \
    augment.name=["ra"] \
    augment.ra.weight="affinity" \
    augment.ra.single="null" \
    augment.ra.init_epoch=20 \
    augment.ra.aff_calc="True" \
    augment.ra.aff_model="/homes/ykohata/code/devml/homes/ypark/code/seg/output/Hflip/weights/best.pth" \
    save.affinity="True" \
    save.affinity_all="True" \
    save.interval=10 \
    && python notify.py 0 || python notify.py 1