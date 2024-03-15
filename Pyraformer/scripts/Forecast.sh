if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

#-------------------------
# ETTm1 & ETTm2
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTm1 \
    -data_path ./ETT-small/ETTm1_ksax.csv \
    -boundaries_df ./ETT-small/ETTm1_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTm1_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTm1_stationary \
    -data_path ./ETT-small/ETTm1_stationary_ksax.csv \
    -boundaries_df ./ETT-small/ETTm1_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTm1_stationary_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTm2 \
    -data_path ./ETT-small/ETTm2_ksax.csv \
    -boundaries_df ./ETT-small/ETTm2_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTm2_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTm2_stationary \
    -data_path ./ETT-small/ETTm2_stationary_ksax.csv \
    -boundaries_df ./ETT-small/ETTm2_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTm2_stationary_$pred_len.log
done
done

#-------------------------
# ETTh1 & ETTh2
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTh1 \
    -data_path ./ETT-small/ETTh1_ksax.csv \
    -boundaries_df ./ETT-small/ETTh1_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTh1_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTh1_stationary \
    -data_path ./ETT-small/ETTh1_stationary_ksax.csv \
    -boundaries_df ./ETT-small/ETTh1_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTh1_stationary_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTh2 \
    -data_path ./ETT-small/ETTh2_ksax.csv \
    -boundaries_df ./ETT-small/ETTh2_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTh2_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data ETTh2_stationary \
    -data_path ./ETT-small/ETTh2_stationary_ksax.csv \
    -boundaries_df ./ETT-small/ETTh2_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_ETTh2_stationary_$pred_len.log
done
done

#-------------------------
# Electricity
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data electricity \
    -data_path ./electricity/electricity_ksax.csv \
    -boundaries_df ./electricity/electricity_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_electricity_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data electricity_stationary \
    -data_path ./electricity/electricity_stationary_ksax.csv \
    -boundaries_df ./electricity/electricity_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_electricity_stationary_$pred_len.log
done
done

#-------------------------
# Traffic
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data traffic \
    -data_path ./traffic/traffic_ksax.csv \
    -boundaries_df ./traffic/traffic_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_traffic_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data traffic_stationary \
    -data_path ./traffic/traffic_stationary_ksax.csv \
    -boundaries_df ./traffic/traffic_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_traffic_stationary_$pred_len.log
done
done

#-------------------------
# weather
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data weather \
    -data_path ./weather/weather_ksax.csv \
    -boundaries_df ./weather/weather_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_weather_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data weather_stationary \
    -data_path ./weather/weather_stationary_ksax.csv \
    -boundaries_df ./weather/weather_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_weather_stationary_$pred_len.log
done
done

#-------------------------
# Illness
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data illness \
    -data_path ./illness/illness_ksax.csv \
    -boundaries_df ./illness/illness_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_illness_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data illness_stationary \
    -data_path ./illness/illness_stationary_ksax.csv \
    -boundaries_df ./illness/illness_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_illness_stationary_$pred_len.log
done
done

#-------------------------
# Exchange Rate
#-------------------------
for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data exchange_rate \
    -data_path ./exchange_rate/exchange_rate_ksax.csv \
    -boundaries_df ./exchange_rate/exchange_rate_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_exchange_rate_$pred_len.log
done
done

for embed_type in 1 2 3 4
do
for pred_len in 12
do
python long_range_main.py \
    -data exchange_rate_stationary \
    -data_path ./exchange_rate/exchange_rate_stationary_ksax.csv \
    -boundaries_df ./exchange_rate/exchange_rate_stationary_ksax_boundaries.csv \
    -embed_type $embed_type \
    -input_size 64 \
    -predict_step $pred_len \
    -inner_size 3 \
    -n_head 6 -n_layer 6 >../logs/LongForecasting/et{$embed_type}_Pyraformer_exchange_rate_stationary_$pred_len.log
done
done
