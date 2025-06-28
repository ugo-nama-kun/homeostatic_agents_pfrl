for i in $(seq 1 2); do
  python main_vision.py --seed $(($i+$2)) --gpu $3 --ident $1
done

