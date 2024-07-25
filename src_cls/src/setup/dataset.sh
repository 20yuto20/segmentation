#!/bin/bash


dataset="cifar10"
HOME_DIR="/homes/ykohata/code/devml/share"
DIR_NAME="dataset"

DATASET_DIR="${HOME_DIR}/${DIR_NAME}"
pwd
mkdir $DATASET_DIR
cd $DATASET_DIR
pwd

if [ "$dataset" == "cifar10" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
elif [ "$dataset" == "cifar100" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    tar -xzvf cifar-100-python.tar.gz
else
    echo "Unknown dataset: $dataset, Select [ cifar10 / cifar100 ]"
fi
    

cd $HOME_DIR
python src/setup/create_dataset.py --dir=$DATASET_DIR --dataset=$dataset


# for abci, compress dir. when using, move to $SGE_LOCALDIR
cd $DATASET_DIR
#tar -czvf cifar10.tar.gz cifar10
# tar -czvf cifar100.tar.gz cifar100