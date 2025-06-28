val=99
for i in $(seq 1 2); do
  python main_thermal.py --reward_setting hs --test --seed $(($val+$i)) --gpu -1 --ident test
done