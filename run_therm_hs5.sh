val=1010
for i in $(seq 1 3); do
  python main_thermal.py --reward_setting hs --seed $(($val+$i+$2)) --gpu $3 --ident $1
done
