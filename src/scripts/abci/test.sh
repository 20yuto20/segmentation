#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=6:00:00
#$ -j y
#$ -o /groups/gaa50073/park-yuna/segmentation/comi/out0714/voc/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.9
source ~/.bashrc
conda activate new

WORKDIR="/groups/gaa50073/park-yuna/segmentation/src/"
echo "ok"

mkdir -p $SGE_LOCALDIR/datasets
cp -v "/groups/gaa50073/park-yuna/datasets/VOCtest_06-Nov-2007.tar" $SGE_LOCALDIR/datasets/
cp -v "/groups/gaa50073/park-yuna/datasets/VOCtrainval_11-May-2012.tar" $SGE_LOCALDIR/datasets/
cd $SGE_LOCALDIR/datasets
ls
# tar -I pigz -xf "VOCtest_06-Nov-2007.tar"
# tar -I pigz -xf "VOCtrainval_11-May-2012.tar"
tar -xf "VOCtest_06-Nov-2007.tar"
tar -xf "VOCtrainval_11-May-2012.tar"
ls
ls VOCdevkit

# Clone the repository and checkout the specified branch
GIT_REPO_DIR="/groups/gaa50073/park-yuna/segmentation"
cd $GIT_REPO_DIR
git fetch origin
git checkout feat-voc-test

cd $WORKDIR

# case $SGE_TASK_ID in
#     1) seed=502;;
#     2) seed=503;;
#     3) seed=504;;
#     *) echo "Invalid task ID"; exit 1;;
# esac
seed=1

python main.py voc_abci \
    default.dataset_dir="$SGE_LOCALDIR/datasets/" \
    default.seed=$seed \
    augment.name=["nan"] 
    