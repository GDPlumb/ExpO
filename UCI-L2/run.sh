export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

rm search.json
rm config.json
rm results_mean.csv
rm results_sd.csv

rm -rf TF-initial
rm -rf TF

python3 run.py
