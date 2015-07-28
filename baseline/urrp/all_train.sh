#!/bin/bash
num=$1
step=$2
begin=$[($num - 1) * $step + 1]
end=$[$num * $step]
i=0
for file in ../../data/reviews_*.gz
do
    if [ -f $file ]; then
        i=$[$i + 1]
        if (($i < $begin)); then
            continue
        elif (($i > $end)); then
            break
        fi
        filename=`basename $file`
        nohup ./urrp -i $file -m 50 -n 50 -s 1 > log/"$filename".log 2>&1 &
    fi
done
