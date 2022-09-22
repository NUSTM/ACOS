BERT_BASE_DIR=/mnt/nfs-storage-titan/BERT/uncased_L-12_H-768_A-12
BASE_DIR=/mnt/nfs-storage-titan/BERT/pytorch_pretrained_BERT
DATA_DIR=$BASE_DIR/ACOS-main/Extract-Classify-ACOS
TASK_NAME=quad
MODEL=quad
DOMAIN=rest16

echo 'DOMAIN is chosen from [rest16, laptop]'
python run_step1.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR\
  --max_seq_length 128 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir $BASE_DIR/output/Extract-Classify-QUAD/${DOMAIN}_1st/


python tokenized_data/get_1st_pairs.py $BASE_DIR $DOMAIN

TASK_NAME=categorysenti
MODEL=categorysenti

python run_step2.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR\
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --output_dir $BASE_DIR/output/Extract-Classify-QUAD/${DOMAIN}_2nd
