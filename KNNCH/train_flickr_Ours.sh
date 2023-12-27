#!/bin/bash
set -e
DEVICE=0
METHOD='NCH'
bits=(16 32 64 128)
alpha=(0.1 0.3 0.5 0.7 0.9)

for i in ${bits[*]}; do
  for j in ${alpha[*]}; do
    echo "**********Start ${METHOD} algorithm**********"
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py --nbit $i \
                                                --dataset flickr \
                                                --epochs 140 \
                                                --alpha_train $j \
                                                --run_times 5
    echo "**********End ${METHOD} algorithm**********"

    echo "**********Start ${METHOD} evaluate**********"
    cd matlab
    matlab -nojvm -nodesktop -r "curve_mean('NCH-Ours', $i, 'flickr', $j, 0.0, 'Ours'); quit;"
    cd ..
    echo "**********End ${METHOD} evaluate**********"
    cd Hashcode
    rm NCH-Ours_flickr*.mat
    cd ..
  done
done

for i in ${bits[*]}; do
 for j in ${alpha[*]}; do
   echo "**********Start ${METHOD} algorithm**********"
   CUDA_VISIBLE_DEVICES=$DEVICE python main.py --nbit $i \
                                               --dataset flickr \
                                               --epochs 140 \
                                               --alpha_query $j \
                                               --run_times 5
   echo "**********End ${METHOD} algorithm**********"

   echo "**********Start ${METHOD} evaluate**********"
   cd matlab
   matlab -nojvm -nodesktop -r "curve_mean('NCH-Ours', $i, 'flickr', 0.0, $j, 'Ours'); quit;"
   cd ..
   echo "**********End ${METHOD} evaluate**********"
   cd Hashcode
   rm NCH-Ours_flickr*.mat
   cd ..
 done
done

for i in ${bits[*]}; do
 for j in ${alpha[*]}; do
   echo "**********Start ${METHOD} algorithm**********"
   CUDA_VISIBLE_DEVICES=$DEVICE python main.py --nbit $i \
                                               --dataset flickr \
                                               --epochs 140 \
                                               --alpha_train $j \
                                               --alpha_query $j \
                                               --run_times 5
   echo "**********End ${METHOD} algorithm**********"

   echo "**********Start ${METHOD} evaluate**********"
   cd matlab
   matlab -nojvm -nodesktop -r "curve_mean('NCH-Ours', $i, 'flickr', $j, $j, 'Ours'); quit;"
   cd ..
   echo "**********End ${METHOD} evaluate**********"
   cd Hashcode
   rm NCH-Ours_flickr*.mat
   cd ..
 done
done