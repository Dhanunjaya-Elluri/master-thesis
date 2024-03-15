# embed_type: 1 2 3 4
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

#------------------------------------------
# ETTm1 & ETTm2
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTm1_ksax.csv\
    --boundaries_df ETT-small/ETTm1_ksax_boundaries.csv\
    --model_id ETTm1_ksax_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTm1_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTm1_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTm1_stationary_ksax.csv\
    --boundaries_df ETT-small/ETTm1_stationary_ksax_boundaries.csv\
    --model_id ETTm1_stationary_$pred_len \
    --model $model_name \
    --data ETTm1_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTm1_stationary_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTm2_ksax.csv\
    --boundaries_df ETT-small/ETTm2_ksax_boundaries.csv\
    --model_id ETTm2_ksax_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTm2_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTm2_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTm2_stationary_ksax.csv\
    --boundaries_df ETT-small/ETTm2_stationary_ksax_boundaries.csv\
    --model_id ETTm2_stationary_$pred_len \
    --model $model_name \
    --data ETTm2_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTm2_stationary_ksax_'$pred_len.log
done
done
done

# #------------------------------------------
# # Electricity
# #------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  electricity/electricity_ksax.csv\
    --boundaries_df electricity/electricity_ksax_boundaries.csv\
    --model_id electricity_ksax_$pred_len \
    --model $model_name \
    --data electricity \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_electricity_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for electricity_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  electricity/electricity_stationary_ksax.csv\
    --boundaries_df electricity/electricity_stationary_ksax_boundaries.csv\
    --model_id electricity_stationary_$pred_len \
    --model $model_name \
    --data electricity_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_electricity_stationary_ksax_'$pred_len.log
done
done
done

#------------------------------------------
# Exchange Rate
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for exchange_rate_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  exchange_rate/exchange_rate_ksax.csv\
    --boundaries_df exchange_rate/exchange_rate_ksax_boundaries.csv\
    --model_id exchange_rate_ksax_$pred_len \
    --model $model_name \
    --data exchange_rate \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_exchange_rate_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for exchange_rate_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  exchange_rate/exchange_rate_stationary_ksax.csv\
    --boundaries_df exchange_rate/exchange_rate_stationary_ksax_boundaries.csv\
    --model_id exchange_rate_stationary_$pred_len \
    --model $model_name \
    --data exchange_rate_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_exchange_rate_stationary_ksax_'$pred_len.log
done
done
done

#------------------------------------------
# Illness
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for illness_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  illness/illness_ksax.csv\
    --boundaries_df illness/illness_ksax_boundaries.csv\
    --model_id illness_ksax_$pred_len \
    --model $model_name \
    --data ili \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_illness_rate_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for illness_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  illness/illness_stationary_ksax.csv\
    --boundaries_df illness/illness_stationary_ksax_boundaries.csv\
    --model_id illness_stationary_$pred_len \
    --model $model_name \
    --data ili_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_illness_stationary_ksax_'$pred_len.log
done
done
done

#------------------------------------------
# Traffic
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for traffic_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  traffic/traffic_ksax.csv\
    --boundaries_df traffic/traffic_ksax_boundaries.csv\
    --model_id traffic_ksax_$pred_len \
    --model $model_name \
    --data traffic \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_traffic_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for traffic_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  traffic/traffic_stationary_ksax.csv\
    --boundaries_df traffic/traffic_stationary_ksax_boundaries.csv\
    --model_id traffic_stationary_$pred_len \
    --model $model_name \
    --data traffic_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_traffic_stationary_ksax_'$pred_len.log
done
done
done

#------------------------------------------
# weather
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for weather_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  weather/weather_ksax.csv\
    --boundaries_df weather/weather_ksax_boundaries.csv\
    --model_id weather_ksax_$pred_len \
    --model $model_name \
    --data weather \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_weather_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for weather_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  weather/weather_stationary_ksax.csv\
    --boundaries_df weather/weather_stationary_ksax_boundaries.csv\
    --model_id weather_stationary_$pred_len \
    --model $model_name \
    --data weather_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_weather_stationary_ksax_'$pred_len.log
done
done
done

#------------------------------------------
# ETTh1 & ETTh2
#------------------------------------------
for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTh1_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTh1_ksax.csv\
    --boundaries_df ETT-small/ETTh1_ksax_boundaries.csv\
    --model_id ETTh1_ksax_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTh1_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTh1_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTh1_stationary_ksax.csv\
    --boundaries_df ETT-small/ETTh1_stationary_ksax_boundaries.csv\
    --model_id ETTh1_stationary_$pred_len \
    --model $model_name \
    --data ETTh1_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTh1_stationary_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTh2_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTh2_ksax.csv\
    --boundaries_df ETT-small/ETTh2_ksax_boundaries.csv\
    --model_id ETTh2_ksax_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTh2_ksax_'$pred_len.log
done
done
done

for embed_type in 1 2 3 4
do
for model_name in Fedformer Autoformer Informer Transformer LogSparse
do
for pred_len in 12
do
  echo "Running $model_name with $embed_type for ETTh2_stationary_ksax_$pred_len"
  python -u main_experiments.py \
    --is_training 1 \
    --root_path ./data/ \
    --data_path  ETT-small/ETTh2_stationary_ksax.csv\
    --boundaries_df ETT-small/ETTh2_stationary_ksax_boundaries.csv\
    --model_id ETTh2_stationary_$pred_len \
    --model $model_name \
    --data ETTh2_stationary \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 3 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1  --embed_type $embed_type >logs/Embedding/$embed_type'_'$model_name'_ETTh2_stationary_ksax_'$pred_len.log
done
done
done
