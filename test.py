import os
import sys
sys.path.append("src/estimator")
import warnings
warnings.filterwarnings("ignore")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# cuda

from transformers import TrainingArguments
import random
import numpy as np
import wandb
import torch
from collections import Counter
from src.utils.callback import ModelMetricCallback

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def main(args):
    global tokenizer
    model, tokenizer = get_model(args)
    model_path = f"./model/{args.load_dir}/pytorch_model.bin"
    print(model_path)
    model.align_model.load_state_dict(torch.load(model_path, map_location=device))
    #model.align_model = model.align_model.to(torch.float32)
    #print(next(model.align_model.parameters()).dtype)
    model = model.to(device)
    model.eval()
    test_ds, _, data_collator = get_dataset(args, tokenizer)


    if args.report_to == 'wandb':
        wandb.init(project='ai_education_Eval', name=f'{args.save_dir}') #Name

    training_args = TrainingArguments(
        output_dir=f"./model/{args.load_dir}",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=0,
        bf16 = args.bf16,
    )

    trainer = LlmTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        data_collator=data_collator,
        callbacks = [
            ModelMetricCallback(args, tokenizer, print_num=20),
        ],
    )
        
    trainer.evaluate()

if __name__=="__main__":
    
    args = get_arguments()
    main(args)