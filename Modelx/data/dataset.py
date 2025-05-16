from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import os
import numpy as np
from typing import Dict, List, Union
import json

class LoRADataset:
    def __init__(self, model_name: str, data_dir: str, split: str = 'train', max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 确保tokenizer有pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.tokenizer.pad_token_id is None:
            raise ValueError("pad_token_id is None")

        # 添加特殊轨迹标记
        special_tokens = {'additional_special_tokens': ['[TRAJ1]', '[TRAJ2]', '[TRAJ3]', '[TRAJ4]']}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # 加载数据集
        json_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{split}.json not found in {data_dir}")
        
        try:
            self.dataset = load_dataset("json", data_files={split: json_path}, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}")

        # 验证第一个样本
        self._validate_sample(self.dataset[0])

    def _validate_sample(self, sample):
        if 'trajectory_features' not in sample or 'poi_id' not in sample:
            raise KeyError("Sample must contain 'trajectory_features' and 'poi_id'")
        if not isinstance(sample['trajectory_features'], list) or len(sample['trajectory_features']) != 4:
            raise ValueError("'trajectory_features' must be a list of 4 trajectories")
        for i, traj in enumerate(sample['trajectory_features']):
            if not isinstance(traj, list) or len(traj) != 1024:
                raise ValueError(f"Trajectory {i} must be a list of 1024 floats")

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        prompt_parts = [
            "Predict the next POI based on trajectories. Target: [TRAJ1]",
            "; Ref1: [TRAJ2]",
            "; Ref2: [TRAJ3]",
            "; Ref3: [TRAJ4]."
        ]
        
        # 合并prompt部分
        prompt = "".join(prompt_parts)
        
        # Tokenize the prompt
        tokenized = self.tokenizer(
            prompt,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=False
        )
        if not tokenized['input_ids'] or tokenized['input_ids'] is None:
            raise ValueError("Failed to tokenize prompt")

        # 获取轨迹标记的位置
        traj_tokens = ['[TRAJ1]', '[TRAJ2]', '[TRAJ3]', '[TRAJ4]']
        traj_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in traj_tokens]
        input_ids = tokenized['input_ids']
        traj_positions = []
        for token_id in traj_token_ids:
            try:
                pos = input_ids.index(token_id)
                traj_positions.append(pos)
            except ValueError:
                raise ValueError(f"Trajectory token {self.tokenizer.convert_ids_to_tokens(token_id)} not found in input_ids")

        batch_inputs = {
            'input_ids': [],
            'attention_mask': [],
            'trajectory_features': [],
            'traj_positions': [],
            'labels': []
        }

        for i in range(len(examples['poi_id'])):
            traj_features = examples['trajectory_features'][i]
            poi_id = examples['poi_id'][i]
            
            batch_inputs['input_ids'].append(tokenized['input_ids'])
            batch_inputs['attention_mask'].append(tokenized['attention_mask'])
            batch_inputs['trajectory_features'].append(traj_features)
            batch_inputs['traj_positions'].append(traj_positions)
            batch_inputs['labels'].append(poi_id)
        
        # 转换为张量
        batch_inputs['input_ids'] = torch.tensor(batch_inputs['input_ids'], dtype=torch.long)
        batch_inputs['attention_mask'] = torch.tensor(batch_inputs['attention_mask'], dtype=torch.long)
        batch_inputs['trajectory_features'] = torch.tensor(batch_inputs['trajectory_features'], dtype=torch.float)  # Shape: [batch_size, 4, 1024]
        batch_inputs['traj_positions'] = torch.tensor(batch_inputs['traj_positions'], dtype=torch.long)  # Shape: [batch_size, 4]
        batch_inputs['labels'] = torch.tensor(batch_inputs['labels'], dtype=torch.long)
        
        return batch_inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.preprocess({
            'trajectory_features': [sample['trajectory_features']],
            'poi_id': [sample['poi_id']]
        })

# 测试代码
if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    
    def create_test_data():
        return {
            "trajectory_features": [
                np.random.rand(1024).round(4).tolist(),
                np.random.rand(1024).round(4).tolist(),
                np.random.rand(1024).round(4).tolist(),
                np.random.rand(1024).round(4).tolist()
            ],
            "poi_id": int(np.random.randint(0, 100))
        }
    
    with open("test_data/train.json", "w") as f:
        for _ in range(2):
            json.dump(create_test_data(), f, ensure_ascii=False)
            f.write("\n")
    
    try:
        train_data = LoRADataset(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            data_dir="test_data",
            split="train"
        )
        sample = train_data[0]
        print("Successfully loaded sample:")
        print(f"Input IDs: {sample['input_ids'].shape}")
        print(f"Attention Mask: {sample['attention_mask'].shape}")
        print(f"Trajectory features: {sample['trajectory_features'].shape}")
        print(f"Trajectory positions: {sample['traj_positions'].shape}")
        print(f"Label: {sample['labels'].shape}")
    except Exception as e:
        print(f"Error: {str(e)}")