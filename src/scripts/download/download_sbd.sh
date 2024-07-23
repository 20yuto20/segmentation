#!/bin/bash

start=`date +%s`

cd ~/segmentaion/datasets/sbd

echo "Downloading SBD dataset..."
curl -LO http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

echo "Extracting SBD dataset..."
tar -xvzf benchmark.tgz

echo "Removing tar file..."
rm benchmark.tgz

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"