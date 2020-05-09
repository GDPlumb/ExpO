
rm -rf trial*
rm *.csv

for VARIABLE in 'trial0' 'trial1' 'trial2' 'trial3' 'trial4' 'trial5' 'trial6' 'trial7' 'trial8' 'trial9'
do
    mkdir $VARIABLE
    cd $VARIABLE
    python ../run.py --train --h_type 'input' --theta_reg_type 'grad3' --theta_reg_lambda 1e-2
    cd ..
done

python agg.py
