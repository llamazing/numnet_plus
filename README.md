# NumNet+

This is the official code repository for NumNet+ (https://leaderboard.allenai.org/drop/submission/bm60vq8f7g2p7t2ld0j0) and NumNet+ v2 (https://leaderboard.allenai.org/drop/submission/bmfuq9e0v32fq8pskug0).
NumNet (https://github.com/ranqiu92/NumNet) was used as a basis for our work.

<img src="numnet_plus_pic.png" alt="Framework" style="zoom:40%;" />

If you use the code, please cite the following paper:

```
@inproceedings{ran2019numnet,
  title={{NumNet}: Machine Reading Comprehension with Numerical Reasoning},
  author={Ran, Qiu and Lin, Yankai and Li, Peng and Zhou, Jie and Liu, Zhiyuan},
  booktitle={Proceedings of EMNLP},
  year={2019}
}
```

## Requirements

`pip install -r requirements.txt`

## Usage
### Data and pretrained roberta-large preparation.
- Download drop data.
  
  `wget -O drop_dataset.zip https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip`
  
  `unzip drop_dataset.zip`

- Download roberta model.
 
  `cd drop_dataset && mkdir roberta.large && cd roberta.large `
  
  `wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin`

- Download roberta config file.
  
  `wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json`
  - Modify `config.json` from `"output_hidden_states": false` to `"output_hidden_states": true`.
  
  
- Download roberta vocab files.
  
  `wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json`
  
  `wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt`  
  
### Train 

- Train with simple multi-span extraction (NumNet+).

    `sh train.sh 345 5e-4 1.5e-5 5e-5 0.01`
    
- Train with tag based multi-span extraction (NumNet+ v2, tag based multi-span paper: http://arxiv.org/abs/1909.13375, github: https://github.com/eladsegal/tag-based-multi-span-extraction).
    
    `sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 tag_mspan`

### Eval
- Save your model as model.pt.
    
    - Simple multi-span extraction (NumNet+).
    
        `sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json`
    
    - Tag based multi-span extraction (NumNet+ v2).
    
        `sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json tag_mspan` 
    
    
    `python drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json`
