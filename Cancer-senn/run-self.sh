
rm -rf self_trial*

for VARIABLE in 'self_trial0' 'self_trial1' 'self_trial2' 'self_trial3' 'self_trial4' 'self_trial5' 'self_trial6' 'self_trial7' 'self_trial8' 'self_trial9'
do
    mkdir $VARIABLE
    cd $VARIABLE
    python ../run-self.py --train --h_type 'input' --theta_reg_type 'grad3' --theta_reg_lambda 1e-2
    cd ..
done

python agg-self.py
