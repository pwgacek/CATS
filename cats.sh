#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=CATS
seq_len=336

# Dataset configurations: name|data_path|dec_in|batch_sizes_per_pred_len
datasets=(
  "air_pollution|air-pollution.csv|8|64,64,64,64"
  "wind_power|wind-power-generation.csv|9|64,128,128,128"
  "microsoft_stock|microsoft-stock.csv|6|64,64,64,64"
  "household_power|household_power_consumption_hourly_clean.csv|7|64,64,64,64"
  "qps|QPS_clean.csv|10|64,64,64,64"
  "sales|sales_clean.csv|8|64,64,64,64"
)

# Prediction lengths and corresponding parameters
pred_lens=(96 192 336 720)
qam_ends=(0.2 0.3 0.5 0.7)

# Function to run CATS for a specific configuration
run_cats() {
  local dataset_name=$1
  local data_path=$2
  local dec_in=$3
  local pred_len=$4
  local qam_end=$5
  local n_heads=$6
  local batch_size=$7
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path $data_path \
    --model_id ${dataset_name}_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --d_layers 3 \
    --dec_in $dec_in \
    --des 'Exp' \
    --itr 1 \
    --d_model 256 \
    --d_ff 512 \
    --n_heads 32 \
    --QAM_end $qam_end \
    --batch_size $batch_size
}

# Main loop
for dataset_config in "${datasets[@]}"; do
  IFS='|' read -r dataset_name data_path dec_in batch_sizes <<< "$dataset_config"
  IFS=',' read -ra batch_size_array <<< "$batch_sizes"
  
  echo "Running CATS on $dataset_name dataset..."
  
  for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    qam_end=${qam_ends[$i]}
    n_heads=${n_heads_list[$i]}
    batch_size=${batch_size_array[$i]}
    
    run_cats "$dataset_name" "$data_path" "$dec_in" "$pred_len" "$qam_end" "$n_heads" "$batch_size"
  done
done

