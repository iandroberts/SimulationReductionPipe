#!/bin/sh

for f in `cat preprocess.txt`
do
    echo $f
    tail -n +64 /home/idroberts/mdpl2/halo_cats/hlist_0.${f}.list > /home/idroberts/mdpl2/processed_halo_cats/hlist_0.${f}_trim.dat
    awk '{print $2, $6, $7, $61}' < /home/idroberts/mdpl2/processed_halo_cats/hlist_0.${f}_trim.dat > /home/idroberts/mdpl2/processed_halo_cats/md0${f}_gaInput.dat
    awk '$61>=10^11 && $61<=10^12.5 {print}' < /home/idroberts/mdpl2/processed_halo_cats/hlist_0.${f}_trim.dat > /home/idroberts/mdpl2/processed_halo_cats/md0${f}_gaMass.dat
    awk '$61>10^12.5 {print}' < /home/idroberts/mdpl2/processed_halo_cats/hlist_0.${f}_trim.dat > /home/idroberts/mdpl2/processed_halo_cats/md0${f}_parentMass.dat
    echo "done"
done
