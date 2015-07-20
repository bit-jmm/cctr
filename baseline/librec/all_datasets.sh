#!/bin/bash
algorithm=$1
num=$2
process_num=$3
pgm=$4
while (( $num < $process_num ))
do
  step=2
  begin=$[($num - 1) * $step + 1]
  end=$[$num * $step]
  i=0
  trainfile=''
  testfile=''
  for file in ../../data/rating_datasets/reviews_*.txt
  do
    if [ -f $file ]; then
      i=$[$i + 1]
      if (($i < $begin)); then
        continue
      elif (($i > $end)); then
        break
      fi
      if [ -z "$testfile" ]; then
        testfile=`basename $file`
      else
        trainfile=`basename $file`
      fi
    fi
  done
  echo "$trainfile : $testfile"
  echo "dataset.ratings.lins=../../data/rating_datasets/$trainfile" > "$algorithm".conf
  echo "ratings.setup=-columns 0 1 2 -threshold -1" >> "$algorithm".conf
  echo "" >> "$algorithm".conf
  echo "recommender=$algorithm" >> "$algorithm".conf
  echo "evaluation.setup=test-set -f ../../data/rating_datasets/$testfile" >> "$algorithm".conf
  echo "item.ranking=off -topN -1 -ignore -1" >> "$algorithm".conf
  echo "" >> "$algorithm".conf
  echo "num.factors=5" >> "$algorithm".conf

  if [ $pgm = 1 ]; then
    echo "num.max.iter=1000" >> "$algorithm".conf
  else
    echo "num.max.iter=30" >> "$algorithm".conf
  fi

  echo "" >> "$algorithm".conf

  if [ $pgm = 1 ]; then
    echo "pgm.setup=-alpha 2 -beta 0.5 -burn-in 500 -sample-lag 100 -interval 100" >> "$algorithm".conf
  else
    echo "learn.rate=0.01 -max -1 -bold-driver" >> "$algorithm".conf
    echo "reg.lambda=0.1 -u 0.1 -i 0.1 -b 0.1 -s 0.001" >> "$algorithm".conf
  fi

  echo "output.setup=off -dir ./demo/Results/ -verbose on" >> "$algorithm".conf

  java -jar librec.jar -c "$algorithm".conf

  num=$[$num + 1]
done
