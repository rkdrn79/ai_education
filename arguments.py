import argparse
from argparse import RawTextHelpFormatter


def get_arguments():
    
    parser = argparse.ArgumentParser(description="ai_educaion", formatter_class=RawTextHelpFormatter)

    parser.add_argument('--is_inference', action='store_true', help='Inference mode')

    #================= parser with data  ===========================#
    parser.add_argument('--dataset', type=str, default='occlusion', help='Dataset') # collision, occlusion
    parser.add_argument('--dataset_use_cache', action='store_true', help='Consider weather to save data and load from cache') # collision, occlusion
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--dataset_load_start', type=int, default=0, help='Number of dataset which will be skipped during loading')
    parser.add_argument('--dataset_load_num', type=int, default=20, help='Dataset load number')

    parser.add_argument('--image_token_len', type=int, default=8, help='image token length / if not used, set 0')
    parser.add_argument('--image_token', type=str, default='<image>', help='image token')

    #================= parser with model  ===========================#
    parser.add_argument('--model_id', type=str, default='google/gemma-2-2b', help='Model id')
    parser.add_argument('--attention_implementation', type=str, default=None, help='Attention implementation')
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer')
    parser.add_argument('--generate_max_new_tokens', type=int, default=128, help='Generate max new tokens')
    parser.add_argument('--z_embed_dim', type=int, default=16, help='Z embedding dimension')

    parser.add_argument('--freeze_llm', action='store_true', help='Freeze LLM')
    parser.add_argument('--freeze_align', action='store_true', help='Freeze Align')

    #================= parser with model optimization  ===========================#
    parser.add_argument('--bf16', action='store_true', help='Bf16')

    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj', help='LoRA target modules')

    parser.add_argument('--packing', action='store_true', help='Use packing, packing is sequence pakcking to reduce memory usage')
    parser.add_argument('--use_reentrant', action='store_true', help='Use reentrant for gradient checkpointing')
    parser.add_argument('--use_flash_attn', action='store_true', help='Use flash attention')

    parser.add_argument('--use_4bit_quantization', action='store_true', help='Use 4-bit quantization')

    #================= parser with train  ===========================#    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Per device eval batch size')
    parser.add_argument('--eval_steps', type=int, default=10, help='Evaluate per {eval_steps} steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save per {save_steps} steps')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--report_to', type=str, default='wandb', help='report to')

    #================= parser with save, load  ===========================#
    parser.add_argument('--save_dir', type=str, default='baseline', help='Save directory')
    parser.add_argument('--load_dir', type=str, default='baseline', help='Load directory')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)