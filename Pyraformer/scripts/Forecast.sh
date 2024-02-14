if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# ETTh1
for pred_len in 48
do
python long_range_main.py -data ETTh1 -input_size 96 -predict_step $pred_len -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_$pred_len.log
done
python long_range_main.py -data ETTh1 -input_size 96 -predict_step 720 -inner_size 5 -n_head 6 >../logs/LongForecasting/Pyraformer_ETTh1_720.log