import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from torch.utils.data import Dataset, DataLoader
from models.lora_model import init_lora_model
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Placeholder for setup_logging (TensorBoard logging)
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    return writer

# Compute metrics with zero_division to suppress warning
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Log metrics to TensorBoard
def log_metrics(writer, metrics, step, prefix='test'):
    for metric_name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{metric_name}", value, step)

class FoursquareDataset(Dataset):
    def __init__(self, json_path, trajectory_dim=128):
        self.data = []
        self.trajectory_dim = trajectory_dim
        self.max_trajectory_count = 4

        with open(json_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'trajectory_features' not in item or 'label' not in item:
                        continue
                    
                    traj = item['trajectory_features']
                    if not isinstance(traj, list) or len(traj) != self.max_trajectory_count:
                        continue
                    for i, sub_traj in enumerate(traj):
                        if not isinstance(sub_traj, list) or len(sub_traj) != self.trajectory_dim:
                            continue
                        if not all(isinstance(x, (int, float)) for x in sub_traj):
                            continue

                    if not isinstance(item['label'], int) or item['label'] < 0:
                        continue

                    seq_length = 512
                    item['input_ids'] = [0] * seq_length
                    item['attention_mask'] = [1] * seq_length
                    item['traj_positions'] = [0, 1, 2, 3]

                    self.data.append(item)
                except json.JSONDecodeError as e:
                    continue

        if self.data:
            print(f"Loaded {len(self.data)} samples from {json_path}")

        for idx, item in enumerate(self.data):
            assert 'input_ids' in item, f"Sample {idx} missing input_ids"
            assert 'attention_mask' in item, f"Sample {idx} missing attention_mask"
            assert 'trajectory_features' in item, f"Sample {idx} missing trajectory_features"
            assert 'traj_positions' in item, f"Sample {idx} missing traj_positions"
            assert 'label' in item, f"Sample {idx} missing label"
            assert isinstance(item['label'], int) and item['label'] >= 0, (
                f"Sample {idx} invalid label: {item['label']}"
            )
            assert isinstance(item['trajectory_features'], list) and len(item['trajectory_features']) == self.max_trajectory_count, (
                f"Sample {idx} trajectory_features length mismatch: {len(item['trajectory_features'])}"
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

def collate_fn(batch, trajectory_dim=128, max_trajectory_count=4):
    try:
        traj_feats = []
        for idx, item in enumerate(batch):
            tf = item['trajectory_features']
            if not isinstance(tf, list) or len(tf) != max_trajectory_count:
                tf = [[0.0] * trajectory_dim] * max_trajectory_count
            elif not all(isinstance(sub_tf, list) and len(sub_tf) == trajectory_dim for sub_tf in tf):
                tf = [[0.0] * trajectory_dim] * max_trajectory_count
            elif not all(all(isinstance(x, (int, float)) for x in sub_tf) for sub_tf in tf):
                tf = [[0.0] * trajectory_dim] * max_trajectory_count
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
            print(f"Batch sample {idx}: {item}")
        raise Exception(f"Collate error: {e}")

def main():
    # Configuration (aligned with train.py)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config_path = "/home/pshao8/poi/Modelx/configs/model_config.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    # Hardcoded configuration (aligned with train.py)
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
            'num_train_epochs': 1,
            'early_stopping_patience': 3
        },
        'model_args': {
            'trajectory_dim': 128,
            'embed_dim': 2048,
            'max_trajectory_count': 4,
            'num_poi_ids': 2880
        },
        'data': {
            'test_dir': "/home/pshao8/poi/Modelx/foursquare_dataset/foursquare_nyc/test_label.json"
        },
        'training': {
            'output_dir': './results'
        }
    }

    # Setup logging
    writer = setup_logging(log_dir=os.path.join(configs['training']['output_dir'], "runs_eval"))

    # Load test dataset
    test_dataset = FoursquareDataset(
        json_path=configs['data']['test_dir'],
        trajectory_dim=configs['model_args']['trajectory_dim']
    )

    # DataLoader
    data_collator = lambda batch: collate_fn(
        batch,
        trajectory_dim=configs['model_args']['trajectory_dim'],
        max_trajectory_count=configs['model_args']['max_trajectory_count']
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=configs['training_args']['batch_size'],
        collate_fn=data_collator,
        shuffle=False
    )

    # Load model
    model = init_lora_model(
        model_name=configs['model']['name'],
        lora_config=configs['lora'],
        trajectory_dim=configs['model_args']['trajectory_dim'],
        embed_dim=configs['model_args']['embed_dim'],
        max_trajectory_count=configs['model_args']['max_trajectory_count'],
        num_poi_ids=configs['model_args']['num_poi_ids']
    )
    checkpoint_path = os.path.join(configs['training']['output_dir'], "best_model", "pytorch_model.bin")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0', weights_only=False))  # Explicitly set weights_only
    model.to('cuda:0')
    model.eval()

    # Evaluation
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'trajectory_features': batch['trajectory_features'],
                'traj_positions': batch['traj_positions'],
                'labels': batch['labels']
            }
            outputs = model(**inputs)
            logits = outputs['logits']
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(pred_ids)
            labels.extend(batch['labels'].cpu().numpy())

    # Compute and log metrics
    metrics = compute_metrics((predictions, labels))
    log_metrics(writer, metrics, 0, prefix='test')

    # Generate report with all metrics, date, and exp_id
    current_date = "2025-05-16"  # Based on provided date
    exp_id = "exp_001"  # Placeholder; replace with your experiment ID or logic
    report = [
        {
            "sample_id": i,
            "predicted_poi_id": int(pred),
            "label_poi_id": int(label),
            "correct": bool(pred == label)  # Convert NumPy bool_ to Python bool
        }
        for i, (pred, label) in enumerate(zip(predictions, labels))
    ]
    report.append({
        "date": current_date,
        "exp_id": exp_id,
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1": float(metrics['f1'])
        }
    })

    with open(os.path.join(configs['training']['output_dir'], "test_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    writer.close()

if __name__ == '__main__':
    main()