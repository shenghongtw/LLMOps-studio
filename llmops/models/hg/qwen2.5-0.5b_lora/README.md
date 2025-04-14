---
library_name: peft
license: other
base_model: Qwen/qwen2.5-0.5b-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: qwen2.5-0.5b_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2.5-0.5b_lora

This model is a fine-tuned version of [Qwen/qwen2.5-0.5b-Instruct](https://huggingface.co/Qwen/qwen2.5-0.5b-Instruct) on the training-data-1744023445 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.15.0
- Transformers 4.50.0
- Pytorch 2.6.0+cpu
- Datasets 3.4.1
- Tokenizers 0.21.0