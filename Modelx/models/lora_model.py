import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os
import json

class LoRAModelWithTrajectory(nn.Module):
    def __init__(self, model_name, lora_config, trajectory_dim, embed_dim, max_trajectory_count, num_poi_ids):
        super().__init__()
        print(f"Loading base model: {model_name}")
        
        # 初始化基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='cuda',
            torch_dtype=torch.float16,
            use_cache=False
        )
        self.base_model.gradient_checkpointing_enable()
        
        # 初始化LoRA配置
        self.lora_config = LoraConfig(
            r=lora_config['rank'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            task_type="CAUSAL_LM"
        )
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        
        # 轨迹特征处理模块
        self.max_trajectory_count = max_trajectory_count
        self.trajectory_projection = nn.Sequential(
            nn.Linear(trajectory_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Linear(embed_dim, num_poi_ids)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 统一移动到GPU
        self.to('cuda:0')
        print(f"Model initialized on {next(self.parameters()).device}")

    def forward(self, input_ids, attention_mask, trajectory_features, traj_positions, labels=None):
        # 设备一致性检查
        device = input_ids.device
        assert all(t.device == device for t in [
            attention_mask, trajectory_features, traj_positions
        ]), "Input tensors must be on the same device"
        
        if labels is not None:
            assert labels.device == device, "Labels must be on the same device"

        with torch.amp.autocast(device_type='cuda'):
            # 1. 处理轨迹特征
            traj_embeds = self.trajectory_projection(trajectory_features.float())
            
            # 2. 处理文本特征
            text_embeds = self.peft_model.base_model.get_input_embeddings()(input_ids)
            
            # 3. 融合特征
            combined_embeds = text_embeds.clone()
            batch_size, seq_len, _ = text_embeds.shape
            
            # 动态插入轨迹特征
            for i in range(batch_size):
                for j, pos in enumerate(traj_positions[i]):
                    if pos < seq_len:
                        combined_embeds[i, pos] = traj_embeds[i, j]
            
            # 4. 模型前向传播
            outputs = self.peft_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 5. 分类头
            cls_hidden = outputs.hidden_states[-1][:, 0]  # 取[CLS] token
            logits = self.classifier(cls_hidden.float())
            
            # 6. 计算损失
            if labels is None:
                return {'logits': logits}
            
            if logits.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Logits shape {logits.shape} != labels shape {labels.shape}"
                )
            
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}

    def save_pretrained(self, save_directory):
        """确保保存完整模型权重（生成pytorch_model.bin）"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存完整模型（包括基础模型+LoRA）
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # 2. 保存配置
        self.peft_model.save_pretrained(save_directory)  # 保存LoRA配置
        
        # 3. 保存自定义模块配置
        custom_config = {
            "trajectory_dim": self.trajectory_projection[0].in_features,
            "embed_dim": self.trajectory_projection[0].out_features,
            "max_trajectory_count": self.max_trajectory_count,
            "num_poi_ids": self.classifier.out_features
        }
        with open(os.path.join(save_directory, "custom_config.json"), "w") as f:
            json.dump(custom_config, f)
        
        print(f"Model saved to {save_directory} with pytorch_model.bin")

def init_lora_model(model_name, lora_config, trajectory_dim, embed_dim, max_trajectory_count, num_poi_ids):
    """初始化模型并验证设备位置"""
    model = LoRAModelWithTrajectory(
        model_name, lora_config, trajectory_dim, embed_dim, max_trajectory_count, num_poi_ids
    )
    
    # 验证所有参数在GPU上
    for name, param in model.named_parameters():
        if not param.is_cuda:
            raise RuntimeError(f"Parameter {name} is not on GPU: {param.device}")
    
    return model