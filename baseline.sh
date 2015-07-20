#!/bin/bash
num=$1
process_num=$2
while (( $num < $process_num ))
do
  step=2
  begin=$[($num - 1) * $step + 1]
  end=$[$num * $step]
  i=0
  trainfile=''
  testfile=''
  for file in ./data/rating_datasets/reviews_*.txt
  do
    if [ -f $file ]; then
      i=$[$i + 1]
      if (($i < $begin)); then
        continue
      elif (($i > $end)); then
        break
      fi
      if [ -z "$testfile" ]; then
        testfile=$file
      else
        trainfile=$file
      fi
    fi
  done
  echo "$trainfile : $testfile"
  ./baseline/MyMediaLite-3.10/bin/rating_prediction --training-file="$trainfile" --test-file="$testfile" --recommender=SVDPlusPlus --recommender-options num_factors=5 --measures=RMSE --no-id-mapping
  num=$[$num + 1]
done
