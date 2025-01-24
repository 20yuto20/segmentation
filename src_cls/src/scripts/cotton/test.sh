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

seed=203

# # For voc
python main.py voc \
    default.seed=$seed \
    learn.n_epoch=50 \
    learn.batch_size=32 \
    augment.name=["ra"] \
    augment.ra.weight="single" \
    augment.ra.single="Solarize" \\
    && python notify.py 0 || python notify.py 1

# # For tiny-imagenet
# python main.py voc \
#     default.seed=$seed \
#     default.dataset_dir="/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/tiny-imagenet-200/" \
#     learn.n_epoch=5 \
#     learn.batch_size=128 \
#     augment.name=["hflip"] \
#     augment.ra.weight="single" \
#     augment.ra.single="hflip" \
#     dataset.name="tiny_imagenet" \
#     dataset.n_class=200 \
#     && python notify.py 0 || python notify.py 1