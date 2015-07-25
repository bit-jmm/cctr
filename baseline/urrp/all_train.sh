#!/bin/bash
num=1
step=5
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
        nohup ./urrp -i $file > ./log/"$filename".log 2>&1 &
    fi
done
