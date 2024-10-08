#!/bin/bash
# Script to download VOC2012 dataset

start=`date +%s`

cd ~

cd segmentaion/datasets/voc

# Download VOC2012
echo "Downloading VOC2012 trainval ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."

# Extract VOC2012 data
echo "Extracting VOC2012 trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar

# Remove tar file
echo "Removing tar file ..."
rm VOCtrainval_11-May-2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"