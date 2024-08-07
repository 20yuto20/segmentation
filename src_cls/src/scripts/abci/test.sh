#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=6:00:00
#$ -j y
#$ -o /groups/gaa50073/park-yuna/kd/comi/out0613/cifar100/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.9
source ~/.bashrc
conda activate new

WORKDIR="/groups/gaa50073/park-yuna/share/src/"
echo "ok"

mkdir -p $SGE_LOCALDIR/datasets
cp -v "/groups/gaa50073/park-yuna/datasets/cifar10.tar.gz" $SGE_LOCALDIR/datasets/
# cp -v "/groups/gaa50073/park-yuna/datasets/cifar100.tar.gz" $SGE_LOCALDIR/datasets/
cd $SGE_LOCALDIR/datasets
tar -I pigz -xf "cifar10.tar.gz"
# tar -I pigz -xf "cifar100.tar.gz"
ls

cd $WORKDIR

seed=501

python main.py cifar10 \
    default.dataset_dir="$SGE_LOCALDIR/datasets/" \
    default.seed=$seed \
    augment.name=["ra"] \
    augment.ra.weight="affinity" \
    learn.n_epoch=4 \
    augment.ra.aff_calc=True \
    augment.ra.aff_model="/groups/gaa50073/park-yuna/share/output/seed501/cifar10/SinglePass0.5_2_Randmag/weights/best.pth"
    