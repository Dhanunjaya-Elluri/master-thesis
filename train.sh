# 0: default
# 1: value embedding + temporal embedding + positional embedding
# 2: value embedding + temporal embedding
# 3: value embedding + positional embedding
# 4: value embedding


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Embedding" ]; then
    mkdir ./logs/Embedding
fi

for embed_type in 1 2 3 4
do
for model_name in Autoformer Informer Transformer Fedformer Logtrans
do 
for pred_len in 96 192 336 720
do
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETTh1_lloyd.csv\
    --model_id ETTh1_lloyd_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTh1_lloyd_'$pred_len.log
done
done
done