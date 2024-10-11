from typing import Dict, List, Tuple, Optional, Any, Union

import os
import wandb
from tqdm import tqdm
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F

from transformers.trainer import Trainer, TRAINING_ARGS_NAME, WEIGHTS_NAME, TrainerCallback
from transformers import AutoTokenizer

class LlmTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)

        target_mask = inputs['target_mask']
        true_tokens = inputs['input_ids']

        if return_outputs:
            label = torch.where(target_mask[:, 1:], true_tokens[:, 1:], torch.zeros_like(target_mask[:, 1:]))
            out = torch.where(target_mask[:, 1:], self.post_process(outputs.logits[:, :-1]), torch.zeros_like(target_mask[:, 1:]))

        target_mask = target_mask[:, 1:].contiguous()
        pred_tokens = outputs.logits[:, :-1].contiguous()[target_mask] # [N, L-1, vocab_size]
        true_tokens = true_tokens[:, 1:].contiguous()[target_mask] # [N, L-1]

        log_probs = -nn.functional.log_softmax(pred_tokens, dim=-1)
        nll_loss = log_probs.gather(dim=-1, index=true_tokens.unsqueeze(-1)).squeeze(-1)
        nll_loss = nll_loss.mean()

        return (nll_loss, out, label) if return_outputs else nll_loss
        
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        model.eval()
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model, inputs, return_outputs=True)

        return (eval_loss, pred, label)

    def post_process(self, logits):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        state_dict = self.model.align_model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
