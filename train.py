import os
import sys
sys.path.append("src/estimator")
import warnings
warnings.filterwarnings("ignore")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import TrainingArguments
import random
import numpy as np
import wandb
import torch
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

from arguments import get_arguments

from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model
from src.trainer import LlmTrainer
from src.utils.callback import ModelMetricCallback

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def main(args):
    ## =================== logging =================== ##

    wandb.init(project=f'AI_Hackathon{args.dataset}', name=f'{args.save_dir}') #Name

    print(f"wandb run name: {wandb.run.name}")

    ## =================== Model =================== ##
    global tokenizer
    model, tokenizer = get_model(args)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        print("adapt lora")

    # Enable flash attention if specified
    if args.use_flash_attn:
        model.llm_model.config.use_flash_attn = True
        print("use flash attn")

    ## =================== Data =================== ##
    train_ds, valid_ds, data_collator = get_dataset(args, tokenizer)

    ## =================== Training =================== ##
    # training arguments
    training_args = TrainingArguments(
        output_dir=f"./model/{args.save_dir}",
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        metric_for_best_model="eval_loss",
        save_strategy="steps",  # 모델 자동 저장 끄기
        save_total_limit=None,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=0,
        # model optimization
        bf16 = args.bf16,
        #packing = args.packing,
        gradient_checkpointing= args.use_reentrant,
        gradient_checkpointing_kwargs= {'use_reentrant': args.use_reentrant},
    )

    # training

    trainer = LlmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        callbacks = [
            ModelMetricCallback(args, tokenizer, print_num=5)
        ]
    )
    
    print(args)
    
    trainer.train()


if __name__=="__main__":
    
    args = get_arguments()
    main(args)