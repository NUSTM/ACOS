
## Running

Modify the corresponding BERT_BASE_DIR, DATA_DIR and output_dir to run the script.

BERT_BASE_DIR: The directory containing config, pytorch_model, and vocab files of BERT (the pytorch BERT model should be added here).

BASE_DIR: The directory of current project.

DATA_DIR: The data directory DIR, where data files are stored at 'DIR/tokenized_data/.' as DOMAIN_YEAR_train.tsv (e.g., rest16_train_quad_bert.tsv).

output_dir: Output directory containing the fine-tuned language model.

## Requirements
* Python 3.7
* Pytorch 1.8
* pytorch-crf 0.7.2

**Running**

Modify the corresponding BERT_BASE_DIR, BASE_DIR, DATA_DIR and output_dir to run the script:
```
sh run.sh
```

