rm -rf TB/*
rm Trials/*
rm results.csv
rm table.csv

python run.py
Rscript ../1-Initial/process.R
