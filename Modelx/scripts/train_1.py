import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from models.lora_model import init_lora_model

from training.trainer import LoRATrainer
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from models.lora_model import init_lora_model
from training.trainer import LoRATrainer
import inspect
print("TrainingArguments loaded from:", inspect.getfile(TrainingArguments))
from transformers import TrainingArguments
print("TrainingArguments loaded from:", TrainingArguments.__module__)
print("TrainingArguments class:", TrainingArguments)

class FoursquareDataset(Dataset):
    def __init__(self, json_path, poi_to_label_map=None, trajectory_dim=128):
        self.data = []
        self.trajectory_dim = trajectory_dim  # 最大轨迹特征长度

        # 初始化 poi_to_label_map
        if poi_to_label_map is None:
            poi_to_label_map = {}
            next_label = 0
        else:
            next_label = len(poi_to_label_map)

        # 读取 JSONL 文件
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'trajectory_features' not in item or 'poi_id' not in item:
                        print(f"跳过缺少必要字段的项: {item}")
                        continue
                    
                    # 将 poi_id 映射到整数 labels
                    poi_id = item['poi_id']
                    if poi_id not in poi_to_label_map:
                        if next_label >= 999:
                            # 超出最大标签统一归为 999 类
                            poi_to_label_map[poi_id] = 999
                        else:
                            poi_to_label_map[poi_id] = next_label
                            next_label += 1
                    item['labels'] = poi_to_label_map[poi_id]

                    # 统一 trajectory_features 长度
                    traj = item['trajectory_features']
                    if len(traj) < self.trajectory_dim:
                        traj += [0.0] * (self.trajectory_dim - len(traj))
                    elif len(traj) > self.trajectory_dim:
                        traj = traj[:self.trajectory_dim]
                    item['trajectory_features'] = traj

                    # 占位符字段
                    seq_length = 512
                    item['input_ids'] = [0] * seq_length
                    item['attention_mask'] = [1] * seq_length
                    item['traj_positions'] = [0, 1, 2, 3]

                    self.data.append(item)
                except json.JSONDecodeError as e:
                    print(f"跳过无效 JSON 行: {e}")
                    continue

        # print(f"从 {json_path} 加载了 {len(self.data)} 个样本")
        # if self.data:
        #     print("样本 0:", {
        #         k: v if k != 'trajectory_features' else v[:5]  # 前5个看一下就好
        #         for k, v in self.data[0].items()
        #     })

        # 数据完整性校验
        for item in self.data:
            assert 'input_ids' in item, f"缺少 input_ids: {item}"
            assert 'attention_mask' in item, f"缺少 attention_mask: {item}"
            assert 'trajectory_features' in item, f"缺少 trajectory_features: {item}"
            assert 'traj_positions' in item, f"缺少 traj_positions: {item}"
            assert 'labels' in item, f"缺少 labels: {item}"
            assert isinstance(item['labels'], int) and 0 <= item['labels'] < 1000, (
                f"无效标签: {item['labels']} 在项: {item}"
            )
            assert len(item['trajectory_features']) == self.trajectory_dim, (
                f"trajectory_features 长度不匹配: {len(item['trajectory_features'])}"
            )

        self.poi_to_label_map = poi_to_label_map

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     return self.data[idx]

        
    def __getitem__(self, idx):
        item = self.data[idx]
        traj = item['trajectory_features']
        #print(item['input_ids'])
        # 直接打印出问题的数据以便诊断
        for i, t in enumerate(traj):
            if not isinstance(t, (float, int)):
                print(f"Non-numeric element found at index {idx}, position {i}: {t}")
            if i==2627:
                pass
            else:
                try:
                    # 可能出错的代码
                    return {
                                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                                "trajectory_features": torch.tensor(t, dtype=torch.float32),
                                "traj_positions": torch.tensor(item["traj_positions"], dtype=torch.long),
                                "labels": torch.tensor(item["labels"], dtype=torch.long)
                            }
                except :print(item['input_ids'])
                    
                

        
       

                




def main():
    # 配置
    import sys
    import transformers
    config_path = "/home/pshao8/poi/Modelx/configs/model_config.yaml"  # 你实际的配置文件路径
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    print("Python path:", sys.executable)
    print("transformers version:", transformers.__version__)
    print("transformers path:", transformers.__file__)
    print("使用的 trajectory_dim 是:", configs["trajectory_dim"])
    print("完整 configs 内容:", configs)

    configs = {
        'model': {
            'name': 'Qwen/Qwen2.5-3B-Instruct'
        },
        'lora': {
            'rank': 16,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj'],
            'dropout': 0.1
        },
        'training_args': {
            'batch_size': 2,
            'learning_rate': 0.0002,
            'num_train_epochs': 3,
            'early_stopping_patience': 3
        },
        'model_args': {
            'trajectory_dim': 128,
            'embed_dim': 2048,
            'max_trajectory_count': 4,
            'num_poi_ids': 1000
        }
    }

    # 初始化模型
    model = init_lora_model(
        model_name=configs['model']['name'],
        lora_config=configs['lora'],
        trajectory_dim=configs['model_args']['trajectory_dim'],
        embed_dim=configs['model_args']['embed_dim'],
        max_trajectory_count=configs['model_args']['max_trajectory_count'],
        num_poi_ids=configs['model_args']['num_poi_ids']
    )

    # 加载数据集
    train_dataset_path = "/home/pshao8/poi/Modelx/foursquare_dataset/foursquare_nyc/train.json"
    val_dataset_path = "/home/pshao8/poi/Modelx/foursquare_dataset/foursquare_nyc/val.json"
    train_dataset = FoursquareDataset(train_dataset_path, trajectory_dim=configs['model_args']['trajectory_dim'])
    val_dataset = FoursquareDataset(val_dataset_path, poi_to_label_map=train_dataset.poi_to_label_map, trajectory_dim=configs['model_args']['trajectory_dim'])

    # 先加载训练集，创建 poi_to_label_map
    #train_dataset = FoursquareDataset(train_dataset_path)
    # 传递 poi_to_label_map 给验证集，确保标签一致
    #val_dataset = FoursquareDataset(val_dataset_path, poi_to_label_map=train_dataset.poi_to_label_map)
    def collate_fn(batch):
        try:
            traj_feats = []
            for idx, item in enumerate(batch):
                tf = item['trajectory_features']
                if not isinstance(tf, list):
                    print(f"[ERROR] 第 {idx} 个 trajectory_features 不是 list，而是 {type(tf)}: {tf}")
                    tf = [0.0] * 1024
                elif not all(isinstance(x, (int, float)) for x in tf):
                    print(f"[ERROR] 第 {idx} 个 trajectory_features 中存在非数值元素: {tf}")
                    tf = [0.0] * 1024
                elif len(tf) != len(batch[0]['trajectory_features']):
                    print(f"[ERROR] 第 {idx} 个 trajectory_features 长度不同：{len(tf)}，应为 {len(batch[0]['trajectory_features'])}")
                    tf = [0.0] * 1024

                traj_feats.append(tf)

            traj_tensor = torch.tensor(traj_feats, dtype=torch.float32)

            # ✅ 检查 NaN / Inf
            if torch.isnan(traj_tensor).any():
                print("[ERROR] trajectory_features 中存在 NaN")
            if torch.isinf(traj_tensor).any():
                print("[ERROR] trajectory_features 中存在 Inf")

            return {
                'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
                'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
                'trajectory_features': traj_tensor,
                'traj_positions': torch.tensor([item['traj_positions'] for item in batch], dtype=torch.long),
                'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
            }
        except Exception as e:
            print("collate_fn 中出现异常:", e)
            print(item['input_ids'])
            raise



    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['training_args']['batch_size'],
        shuffle=True,
        # collate_fn=collate_fn
    )

    # 调试 DataLoader
    for batch in train_dataloader:
        print("批次键:", batch.keys())
        print("标签:", batch.get('labels'))
        print("轨迹特征形状:", batch['trajectory_features'].shape)
        break

    # 训练参数
    # training_args = TrainingArguments(
    #     output_dir='./results',
    #     per_device_train_batch_size=configs['training_args']['batch_size'],
    #     per_device_eval_batch_size=configs['training_args']['batch_size'],
    #     learning_rate=configs['training_args']['learning_rate'],
    #     num_train_epochs=configs['training_args']['num_train_epochs'],
    #     gradient_accumulation_steps=1,
    #     logging_steps=100,
    #     load_best_model_at_end=True,
    #     eval_steps=500,  # 评估的步骤
    #     save_steps=500,  # 保存的步骤
    #     evaluation_strategy="steps",  # 强制设置为 steps
    #     save_strategy="steps",       # 强制设置为 steps
    # )
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        save_strategy='steps',
        #evaluation_strategy="steps",   # 评估策略设置为 steps
        #save_strategy="steps",         # 保存策略设置为 steps
        save_steps=1000,               # 每1000步保存一次
        eval_steps=1000,               # 每1000步评估一次
        load_best_model_at_end=True,   # 加载最好的模型
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=500,
        num_train_epochs=3,
        # 其他配置
    )




    # 初始化训练器
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        early_stopping_patience=configs['training_args']['early_stopping_patience']
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    
    main()