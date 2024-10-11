import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import os
import json

class ImageTokenModel(nn.Module):
    def __init__(self, image_size, patch_size, z_embed_dim, hidden_dim, object_token_len):
        super(ImageTokenModel, self).__init__()
        self.image_size = image_size  # H, W tuple
        self.patch_size = patch_size  # h, w tuple
        self.z_embed_dim = z_embed_dim
        self.hidden_dim = hidden_dim
        self.object_token_len = object_token_len

        # Image patch embedding
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=z_embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Linear(self.num_patches, hidden_dim)
        )

        self.learnable_query = nn.Parameter(torch.randn(1, object_token_len, hidden_dim), requires_grad=True)
        nn.init.normal_(self.learnable_query, mean=0, std=1 / hidden_dim)

        # Use only one MultiheadAttention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs):
        """
        inputs: [B, C, H, W] - Image batch
        """
        batch_size = inputs.shape[0]

        # Convert image to patch embeddings
        patch_emb = self.patch_emb(inputs)  # [B, hidden_dim, num_patches]
        patch_emb = einops.rearrange(patch_emb, 'b c n -> b n c')  # [B, num_patches, hidden_dim]

        query = self.learnable_query.repeat(batch_size, 1, 1)

        # Single Multihead Attention Layer
        out, _ = self.cross_attn(query, patch_emb, patch_emb)
        query = self.norm(out + query)

        out = self.final_layer(query)

        out = out.view(-1, out.size(-1))
        return out

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_weights_path)

        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "z_embed_dim": self.z_embed_dim,
            "hidden_dim": self.hidden_dim,
            "object_token_len": self.object_token_len,
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file)

        print(f"Model weights and configuration saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory):
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        
        model = cls(**config)

        model_weights_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_weights_path))

        return model