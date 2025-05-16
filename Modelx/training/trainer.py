from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from torch import amp
import torch
import os

class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        early_stopping_patience = kwargs.pop('early_stopping_patience', None)
        super().__init__(*args, **kwargs)

        if early_stopping_patience is not None:
            self.add_callback(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.01
            ))

        self.scaler = amp.GradScaler()
        self.move_model_to_device(self.model, torch.device("cuda:0"))

    def move_model_to_device(self, model, device):
        model.to(device)
        for name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(device)
        for name, param in model.named_parameters():
            param.data = param.data.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model output does not contain 'loss' key.")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with amp.autocast(device_type="cuda"):
            loss = self.compute_loss(model, inputs)

        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Expected loss to be a Tensor, but got {type(loss)}.")

        self.scaler.scale(loss).backward()

        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        if self.state.global_step % 100 == 0:
            print(f"Step {self.state.global_step}: Loss = {loss.item()}")

        return loss.detach()

    def evaluation_step(self, model, inputs, return_outputs=False, **kwargs):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with amp.autocast(device_type="cuda"):
            outputs = model(**inputs)

        return outputs

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 直接调用模型的 save_pretrained（已在 lora_model.py 中实现）
        self.model.save_pretrained(output_dir)
        
        # 保存训练状态（优化器、调度器等）
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),  # Fixed from scheduler to lr_scheduler
            },
            os.path.join(output_dir, "training_state.bin")
        )
        print(f"Model saved to {output_dir}")

    def train(self, *args, **kwargs):
        # Call parent train method
        output = super().train(*args, **kwargs)
        
        # After training, if load_best_model_at_end=True, save the loaded best model
        if self.args.load_best_model_at_end:
            best_model_dir = os.path.join(self.args.output_dir, "best_model")
            self.save_model(output_dir=best_model_dir)
            print(f"Best model saved to {best_model_dir} at the end of training")
        
        return output