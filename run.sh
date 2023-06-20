#! /bin/bash

CUDA_DEVICE=1
CACHE_DIR=/home/hywen/BERT_CACHE
BATCH_SIZE=16
UPDATE_BATCH_SIZE=32
LR=5e-6
EPOCHS=30

DIR_SUFFIX=bart_template_post_${LR}_b_${UPDATE_BATCH_SIZE}_wiki_predict_topic_stance_neg_topic_neg
LOG_DIR=./logs/$DIR_SUFFIX
mkdir $LOG_DIR

export CUDA_VISIBLE_DEVICES\=$CUDA_DEVICE 

for SEED in 0 1 2 3 4
do
    OUTPUT_DIR=/mnt/zero_shot_stance/${DIR_SUFFIX}_seed_$SEED
    nohup python train.py \
        --model_name bart-base \
        --model_type bart_template \
        --cache_dir $CACHE_DIR \
        --batch_size $BATCH_SIZE \
        --update_batch_size $UPDATE_BATCH_SIZE \
        --lr $LR \
        --num_train_epochs $EPOCHS \
        --do_train \
        --output_dir $OUTPUT_DIR \
        --wiki_path data/VAST/wiki_dict.pkl \
        --predict_topic \
        --predict_stance_neg \
        --predict_topic_neg \
        --seed $SEED > ${LOG_DIR}/seed_${SEED}.log
done