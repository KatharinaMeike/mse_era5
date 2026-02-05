#!/bin/bash
year=$1
monthstart=$2
monthend=$3
monthstep=$4

for (( i=$month_start; i<=$monthend; i=i+$monthstep )); do
    ms=$i; me=$(($i + $monthstep-1))
    if [ ${me} -gt $monthend ]
    then
        ye=$monthend
    fi
    echo "Month $year ${ms} - ${me}"
    folder="$year_${ms}_${me}"
    mkdir "$folder"
    if [ ${ms} -lt 10 ]
    then
      sed -e "s/_SD_/0${ms}/g" < energybudget_KW_lonlev_lonlat_template.py >  "$folder/energybudget_KW.py"
    else
      sed -e "s/_SD_/${ys}/g" < energybudget_KW_lonlev_lonlat_template.py >  "$folder/energybudget_KW.py"
    fi
    if [ ${ye} -lt 10 ]
    then
      sed -i "s/_ED_/0${ye}/g" "$folder/energybudget_KW.py"
    else
      sed -i "s/_ED_/${ye}/g" "$folder/energybudget_KW.py"
    fi
    sed -i "s/_MM_/$4/g" "$folder/energybudget_KW.py"
    cd "$folder"
    pwd
    ls  # submit job here 
    sbatch run_energybudget_KW.sh
    cd ..
    sleep 5s
done
