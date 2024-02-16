if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# ETTh1
for pred_len in 12
do
python long_range_main.py \
    -data ETTh1 \
    -input_size 48 \
    -predict_step $pred_len \
    -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
python long_range_main.py \
    -data electricity \
    -data_path ./electricity/electricity_lloyd.csv \
    -boundaries_df ./electricity/electricity_lloyd_boundaries.csv \
    -input_size 48 \
    -predict_step $pred_len \
    -inner_size 5 \
    -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
done
