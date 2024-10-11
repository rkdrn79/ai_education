import torch
import numpy as np
import jax
import jax.numpy as jnp

class LLMDataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, features):
        # Extract the features
        input_ids = [torch.tensor(feature["input_ids"]) for feature in features]
        input_ids = torch.stack(input_ids).to(torch.long)
        
        attention_mask = [torch.tensor(feature["attention_mask"]) for feature in features]
        attention_mask = torch.stack(attention_mask).to(torch.long)

        token_type_ids = [torch.tensor(feature["token_type_ids"]) for feature in features]
        token_type_ids = torch.stack(token_type_ids).to(torch.long)

        target_mask = [torch.tensor(feature["target_mask"]) for feature in features]
        target_mask = torch.stack(target_mask).to(torch.bool)

        iamge = [torch.tensor(feature["image"]) for feature in features]
        image = torch.stack(image)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # oricorn values are added to the return dictionary
            'image': image,
            'token_type_ids': token_type_ids,
            'target_mask': target_mask
        }
    