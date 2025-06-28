val=10
for i in $(seq 1 3); do
  python main_trp.py --reward_setting hs --seed $(($val+$i+$2)) --gpu $3 --ident $1
done
