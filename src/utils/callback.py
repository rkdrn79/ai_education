import torch
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from torch.cuda.amp import autocast

class ModelMetricCallback(TrainerCallback):
    def __init__(self, args, tokenizer: AutoTokenizer, print_num = 0):
        self.args = args
        self.tokenizer = tokenizer
        self.print_num = print_num
        if args.report_to == 'wandb':
            import wandb
            self._wandb = wandb

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    @staticmethod
    def make_generation_input(inputs, pad_token_id):
        input_ids = inputs['input_ids']

        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        target_mask = inputs['target_mask']
        answer_ids = input_ids.clone()
        answer_ids[~target_mask] = pad_token_id

        question_ids = input_ids
        question_ids[target_mask] = pad_token_id
        attention_mask[target_mask] = 0

        tm_counts = target_mask.sum(dim=1)
        question_ids = torch.stack([
            question_id.roll(tm.item())
            for question_id, tm in zip(question_ids, tm_counts)
        ])
        token_type_ids = torch.stack([
            token_type_id.roll(tm.item())
            for token_type_id, tm in zip(token_type_ids, tm_counts)
        ])
        attention_mask = torch.stack([
            am.roll(tm.item())
            for am, tm in zip(attention_mask, tm_counts)
        ])

        inputs = {
            **inputs,
            'token_type_ids': token_type_ids,
            'input_ids': question_ids,
            'attention_mask': attention_mask,
        }
        return inputs, question_ids, answer_ids


    # TODO - simplify / refactor this
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        if not state.is_world_process_zero:
            return

        questions = []
        answer_trues = []
        answer_preds = []
        eval_num = 0

        for inputs in eval_dataloader:
            batch_size, seq_len = inputs['input_ids'].shape
            inputs, question_ids, answer_ids = self.make_generation_input(inputs, self.tokenizer.pad_token_id)

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
                    pred_ids = model.generate(inputs)[:, seq_len:]
            questions.extend(self.tokenizer.batch_decode(question_ids, skip_special_tokens=True))
            answer_preds.extend(self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            answer_trues.extend(self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True))
            eval_num += batch_size

        for i, (question, answer_true, answer_pred) in enumerate(zip(questions, answer_trues, answer_preds)):
            if i == self.print_num:
                break
            print(f"instance : {i}")
            print("======================================================================================")
            print(f"Question: {question}")
            print(f"True Answer: {answer_true}")
            print(f"Pred Answer: {answer_pred}")
            print("======================================================================================")

        return questions, answer_trues, answer_preds
    
