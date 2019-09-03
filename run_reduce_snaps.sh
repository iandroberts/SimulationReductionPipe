#!/bin/sh

for f in `cat snaplist.txt`
do
    echo $f
    tail -n +64 /home/idroberts/mdpl2/halo_cats/hlist_0.${f}.list > /home/idroberts/mdpl2/processed_halo_cats/hlist_0.${f}_trim.dat
done
