
## Running

Modify the corresponding BERT_BASE_DIR, DATA_DIR and output_dir to run the script.

BERT_BASE_DIR: The directory containing config, pytorch_model, and vocab files of BERT (the pytorch BERT model should be added here).

DATA_DIR: The data directory DIR, where data files are stored at 'DIR/data/.' as DOMAIN_YEAR_train.tsv (e.g., rest16_train_quad_bert.tsv).

output_dir: Output directory containing the fine-tuned language model.

**Step 1:**

Extract aspects and opinions from review text:
```
sh run_step1.sh
```

where the run_step1.sh (in fold step1) script is:

```
export BERT_BASE_DIR=/home/hjcai/8RTX/BERT/uncased_L-12_H-768_A-12
export DATA_DIR=/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad
export TASK_NAME=quad
export MODEL=quad
export DOMAIN=laptop

echo $BERT_BASE_DIR
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
  --num_train_epochs 30.0 \
  --output_dir /home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/output/Extract-Classify4Quad/laptop
```

**Step 2:**

Delete the arg "do_train" in the run_step1.sh to get the predicted aspects and opinions on test dataset.

**run the get_1st_pair.py in "./data" to get the input test examples.** 

Perform category-sentiment classification given candidate aspect-opinion pairs:
```
sh run_ext.sh
```

where the run_ext.sh (in fold step2) script is :

```
export BERT_BASE_DIR=/home/hjcai/8RTX/BERT/uncased_L-12_H-768_A-12
export DATA_DIR=/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd
export TASK_NAME=categorysenti
export MODEL=categorysenti
export DOMAIN=laptop

python run_classifier_step2.py \
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
  --num_train_epochs 17.0 \
  --output_dir /home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/output/Extract-Classify4Quad-2nd/$DOMAIN/laptop
```
