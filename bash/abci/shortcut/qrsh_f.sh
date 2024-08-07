#!/bin/bash
#$ -g gaa50073
#$ -l rt_F=1
#$ -l h_rt=01:00:00

group_name="#$ -g gaa50073"
resource_type="#$ -l rt_F=1"
execution_time="#$ -l h_rt=01:00:00"

# echoでオプションを出力
echo "run qrsh-f with following conditions"
echo "----------"
echo "$group_name"
echo "$resource_type"
echo "$execution_time"
echo "----------"

qrsh -g gaa50073 -l rt_F=1 -l h_rt=01:00:00

