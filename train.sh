# Training script
python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

# To run this script, use the command
# bash train.sh