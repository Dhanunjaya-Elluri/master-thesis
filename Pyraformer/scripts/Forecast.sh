if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# ETTh1
# for pred_len in 12
# do
# python long_range_main.py \
#     -data ETTh1 \
#     -input_size 48 \
#     -predict_step $pred_len \
#     -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
# python long_range_main.py \
#     -data electricity \
#     -data_path ./electricity/electricity_lloyd.csv \
#     -boundaries_df ./electricity/electricity_lloyd_boundaries.csv \
#     -input_size 48 \
#     -predict_step $pred_len \
#     -inner_size 5 \
#     -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
# done

for embed_type in 1 2 3 4
do
for pred_len in 32
do
# python long_range_main.py \
#     -data ETTh1 \
#     -input_size 32 \
#     -predict_step $pred_len \
#     -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
python long_range_main.py \
    -data ETTh1_stationary \
    -data_path ./ETT-small/ETTh1_lloyd_stationary.csv \
    -boundaries_df ./ETT-small/ETTh1_lloyd_stationary_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTh1_$pred_len.log
done
done
