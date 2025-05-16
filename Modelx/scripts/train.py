import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from models.lora_model import init_lora_model
from training.trainer import LoRATrainer
import yaml

class FoursquareDataset(Dataset):
    def __init__(self, json_path, trajectory_dim=128):
        self.data = []
        self.trajectory_dim = trajectory_dim  # 每个轨迹的特征维度
        self.max_trajectory_count = 4  # 固定 4 个轨迹

        # 读取 JSONL 文件
        with open(json_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'trajectory_features' not in item or 'label' not in item:
                        continue
                    
                    # 验证 trajectory_features 格式
                    traj = item['trajectory_features']
                    if not isinstance(traj, list) or len(traj) != self.max_trajectory_count:
                        continue
                    for i, sub_traj in enumerate(traj):
                        if not isinstance(sub_traj, list) or len(sub_traj) != self.trajectory_dim:
                            continue
                        if not all(isinstance(x, (int, float)) for x in sub_traj):
                            continue

                    # 确保 label 是整数且有效
                    if not isinstance(item['label'], int) or item['label'] < 0:
                        continue

                    # 占位符字段
                    seq_length = 512
                    item['input_ids'] = [0] * seq_length
                    item['attention_mask'] = [1] * seq_length
                    item['traj_positions'] = [0, 1, 2, 3]

                    self.data.append(item)
                except json.JSONDecodeError as e:
                    continue

        if self.data:
            print()

        # 数据完整性校验
        for idx, item in enumerate(self.data):
            assert 'input_ids' in item, f"样本 {idx} 缺少 input_ids"
            assert 'attention_mask' in item, f"样本 {idx} 缺少 attention_mask"
            assert 'trajectory_features' in item, f"样本 {idx} 缺少 trajectory_features"
            assert 'traj_positions' in item, f"样本 {idx} 缺少 traj_positions"
            assert 'label' in item, f"样本 {idx} 缺少 label"
            assert isinstance(item['label'], int) and item['label'] >= 0, (
                f"样本 {idx} 无效标签: {item['label']}"
            )
            assert isinstance(item['trajectory_features'], list) and len(item['trajectory_features']) == self.max_trajectory_count, (
                f"样本 {idx} trajectory_features 长度不匹配: {len(item['trajectory_features'])}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'trajectory_features': item['trajectory_features'],
            'traj_positions': item['traj_positions'],
            'labels': item['label']
        }

def main():
    # 配置
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config_path = "/home/pshao8/poi/Modelx/configs/model_config.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    # 硬编码配置（确保与 YAML 一致）
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
            'batch_size': 64,
            'learning_rate': 0.0005,
            'num_train_epochs': 20,
            'early_stopping_patience': 3
        },
        'model_args': {
            'trajectory_dim': 128,
            'embed_dim': 2048,
            'max_trajectory_count': 4,
            'num_poi_ids': 2880
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
    ).to('cuda:0')

    # 加载数据集
    train_dataset_path = "/home/pshao8/poi/Modelx/foursquare_dataset/foursquare_nyc/train_label.json"
    val_dataset_path = "/home/pshao8/poi/Modelx/foursquare_dataset/foursquare_nyc/val_label.json"
    train_dataset = FoursquareDataset(
        train_dataset_path,
        trajectory_dim=configs['model_args']['trajectory_dim']
    )
    val_dataset = FoursquareDataset(
        val_dataset_path,
        trajectory_dim=configs['model_args']['trajectory_dim']
    )

    def collate_fn(batch):
        try:
            # 验证 trajectory_features 格式
            traj_feats = []
            for idx, item in enumerate(batch):
                tf = item['trajectory_features']
                if not isinstance(tf, list) or len(tf) != configs['model_args']['max_trajectory_count']:
                    tf = [[0.0] * configs['model_args']['trajectory_dim']] * configs['model_args']['max_trajectory_count']
                elif not all(isinstance(sub_tf, list) and len(sub_tf) == configs['model_args']['trajectory_dim'] for sub_tf in tf):
                    tf = [[0.0] * configs['model_args']['trajectory_dim']] * configs['model_args']['max_trajectory_count']
                elif not all(all(isinstance(x, (int, float)) for x in sub_tf) for sub_tf in tf):
                    tf = [[0.0] * configs['model_args']['trajectory_dim']] * configs['model_args']['max_trajectory_count']
                traj_feats.append(tf)

            return {
                'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long).cuda(),
                'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long).cuda(),
                'trajectory_features': torch.tensor(traj_feats, dtype=torch.float32).cuda(),
                'traj_positions': torch.tensor([item['traj_positions'] for item in batch], dtype=torch.long).cuda(),
                'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long).cuda()
            }
        except Exception as e:
            for idx, item in enumerate(batch):
                print()
            raise

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['training_args']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 调试 DataLoader
    try:
        for batch in train_dataloader:
            break
    except Exception as e:
        print()
        raise

    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        save_strategy='steps',
        save_steps=20,
        eval_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',  # Explicitly specify metric for best model
        greater_is_better=False,  # Lower eval_loss is better
        per_device_train_batch_size=configs['training_args']['batch_size'],
        per_device_eval_batch_size=configs['training_args']['batch_size'],
        logging_dir='./logs/0516_1',
        logging_steps=20,
        num_train_epochs=configs['training_args']['num_train_epochs'],
        learning_rate=configs['training_args']['learning_rate']
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