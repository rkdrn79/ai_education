import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.model.models.llm_model import LLMModel
from src.model.models.image_bridges import ImageTokenModel
from src.utils.tokentype import TokenType
import importlib

def get_model(args):
    # Define the LLM model
    if args.use_4bit_quantization:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation=args.attention_implementation,
        cache_dir=args.huggingface_cache_dir,
        quantization_config=bnb_config if args.use_4bit_quantization else None,
    )
    
    # Define the tokenizer
    if args.tokenizer is None:
        args.tokenizer = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        TOKENIZERS_PARALLELISM=False,
        cache_dir=args.huggingface_cache_dir,
        trust_remote_code=False,
    )
    
    tokenizer.padding_side  = 'left'
    
    added_tokens = ['<image>']

    # Add new tokens to tokenizer
    for token in added_tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens(token)
    # Resize the model's token embeddings to accommodate new tokens
    llm_model.resize_token_embeddings(len(tokenizer))

    image_bridge = ImageTokenModel(
        hidden_dim=llm_model.config.hidden_size,
        object_token_len = args.object_token_len,
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        z_embed_dim=args.z_embed_dim
    )
    
    # Define the LLMPlaner model
    llm_planer_model = LLMModel(args, llm_model, align_model={TokenType.IMAGE: image_bridge})#, quantization_config=quantization_config)
    llm_planer_model.freeze(llm_model = args.freeze_llm, align_model = args.freeze_align)

    print(f"Model has {count_parameters(llm_planer_model.llm_model)} parameters.")
    print(f"Align model has {count_parameters(llm_planer_model.align_model)} parameters.")

    return llm_planer_model, tokenizer

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params