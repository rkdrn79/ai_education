import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.utils.tokentype import TokenType

class ScaleEmbed(nn.Module):
    def __init__(self, embed, scale):
        super(ScaleEmbed, self).__init__()
        self.embed = embed
        self.scale = scale

    def forward(self, x):
        return self.embed(x) * torch.tensor(self.scale, dtype=x.dtype, device=x.device)

class LLMModel(nn.Module):
    def __init__(self, args, llm_model = None, align_model = {TokenType.IMAGE: None}):
        super(LLMModel, self).__init__()
        self.args = args
        
        # Define the llm model
        self.llm_model = llm_model
        self.config = self.llm_model.model.config

        self.align_model = nn.ModuleDict({
            TokenType.to_str(key): value
            for key, value in align_model.items() if value is not None
        })

        if "google/gemma" in self.llm_model.model.config.model_type:
            # if model is gemma 2-2b, scale the embeddings by the square root of the hidden dimension. We must only do this for embed_tokens layer not embed from bridge.
            # Empirically it hurts the bridge performance if we scale the embeddings from bridge.
            hidden_dim = self.llm_model.model.config.hidden_size
            self.llm_model.model.embed_tokens = ScaleEmbed(
                self.llm_model.model.embed_tokens,
                hidden_dim ** 0.5,
            )
            self.llm_model.model.config.hidden_size = 1

    def forward(self, inputs):
        inputs_embeds, attention_mask = self.token_match(inputs), inputs['attention_mask']
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return outputs

    def generate(self, inputs):
        inputs_embeds, attention_mask = self.token_match(inputs), inputs['attention_mask']

        outputs = self.llm_model.generate(
            input_ids = inputs['input_ids'],
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = self.args.generate_max_new_tokens,
            past_key_values = None,
        )

        return outputs

    def token_match(self, inputs):
        # Extract input tensors
        token_type_ids = inputs['token_type_ids']
        inputs_embeds = self.llm_model.model.embed_tokens(inputs['input_ids'])
        for key in self.align_model.keys():
            if self.args.bf16:
                inputs_embeds[token_type_ids == TokenType.to_int(key)] = self.align_model[key](inputs).to(torch.bfloat16)
            else:
                inputs_embeds[token_type_ids == TokenType.to_int(key)] = self.align_model[key](inputs)

        return inputs_embeds
    
    def freeze(self, llm_model = False, align_model = False):
        if llm_model:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        if align_model:
            for param in self.align_model.parameters():
                param.requires_grad = False
