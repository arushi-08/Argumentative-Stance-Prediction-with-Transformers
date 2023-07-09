CUDA_PATH=/home/kundank/cuda_libraries/cuda-11.6.0
CUDA_DEVICES=2,3,4,5,6,7
MODEL_SIZE=large
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
MAX_SOURCE_LENGTH=8192
MAX_TARGET_LENGTH=300
DATASET_PREFIX=factual_editor/for_flant5/all_0.03_0.03
NUM_EPOCHS=10
GRADIENT_ACCUMULATION_STEPS=32
OUTPUT_ROOT=ser_dirs


all_tasks=(
evidence_extraction_with_fixfactuality_nofulldelete
)

for TASK in "${all_tasks[@]}"
do
echo ${TASK}
RUN_NAME=`echo $RANDOM | md5sum | head -c 10; echo;`
CUDA_HOME=$CUDA_PATH deepspeed --include=localhost:${CUDA_DEVICES} --master_port 48992 examples/pytorch/summarization/run_summarization.py \
  --model_name_or_path google/flan-t5-${MODEL_SIZE} --do_train --do_eval \
  --output_dir $OUTPUT_ROOT/${TASK}_flant5_${MODEL_SIZE} \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --max_train_samples 999999 --max_eval_samples 999999 --max_predict_samples 999999 \
  --max_source_length ${MAX_SOURCE_LENGTH} --max_target_length ${MAX_TARGET_LENGTH} \
  --deepspeed tests/deepspeed/t0ds_config.json \
  --gradient_checkpointing \
  --train_file /home/kundank/dataset_root/${DATASET_PREFIX}/${TASK}/train.jsonl \
  --validation_file /home/kundank/dataset_root/${DATASET_PREFIX}/${TASK}/validation.jsonl \
  --evaluation_strategy epoch \
  --num_train_epochs ${NUM_EPOCHS} --save_strategy epoch --text_column input_string --summary_column  output_string \
  --report_to wandb --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_total_limit 2 --load_best_model_at_end \
  --run_name $RUN_NAME --metric_for_best_model eval_loss --greater_is_better false --learning_rate 5e-5

echo $RUN_NAME > $OUTPUT_ROOT/${TASK}_flant5_${MODEL_SIZE}/wandb_runid.txt

done
