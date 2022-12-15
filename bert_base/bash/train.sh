# !/bin/bash

BERT_BASE_DIR=D:\pyWS\bert_base\pre_model\bert-base-uncased
BERT_ALBERT_DIR=D:\pyWS\bert_base\pre_model\albert-.....

DATA_DIR=D:\pyWS\bert_base\tiktok
OUTPUT_DIR=D:\pyWS\bert_base\output\20221212_bert_output
ADD_DATA_PATH=

LEARNING_RATE=8e-5

MODEL_TYPE=bert_catcls

python3 run.py\
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --batch_size_rdrop=1 \
    --data_dir=$DATA_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$BERT_BASE_DIR \
    --max_seq_length=64 \
    --per_gpu_train_batch=256 \
    --per_gpu_eval_batch=512 \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=100.0 \
    --output_dir=$OUTPUT_DIR \
    --overwrite_output_dir \
    --logging_steps=200 \
    --save_steps=200 \
    --model_method="Align" \
    --do_lower_case \
    --keep_checkpoint_max=3 \
    --add_data_path=$ADD_DATA_PATH