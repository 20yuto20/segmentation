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

# seed=10001
# python main.py voc \
#     default.seed=$seed \
#     learn.n_epoch=1 \
#     learn.batch_size=32 \
#     augment.name=["ra"] \
#     augment.ra.weight="random" \
#     augment.ra.single="SolarizeAdd" \
#     && python notify.py 0 || python notify.py 1

seed=2024
python main.py voc \
    default.device_id=0 \
    learn.n_epoch=50 \
    learn.batch_size=32 \
    default.seed=$seed \
    default.num_workers=10 \
    augment.name=["ra"] \
    augment.ra.weight="affinity" \
    augment.ra.single="null" \
    augment.ra.init_epoch=20 \
    augment.ra.aff_calc="True" \
    augment.ra.aff_model="/homes/ykohata/code/devml/homes/ypark/code/seg/output_cls/aff_weight/seed2024/RA2_Random_Randmag/weights/best.pth" \
    save.affinity="True" \
    save.affinity_all="True" \
    save.interval=10 \
    && python notify.py 0 || python notify.py 1