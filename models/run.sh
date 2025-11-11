for i in 1e-3 5e-4 1e-4
do
    python ./K-Radar/models/GS.py --resolution 32,32,32 --lr "$i" >> ./K-Radar/models/logs/${i}.txt
done
