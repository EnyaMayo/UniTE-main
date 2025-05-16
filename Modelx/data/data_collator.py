from transformers import DataCollatorWithPadding
import torch

def get_data_collator(tokenizer):
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def collator(features):
        text_features = [
            {
                'input_ids': f['input_ids'].squeeze(0),
                'attention_mask': f['attention_mask'].squeeze(0),
                'labels': f['labels'].squeeze(0)
            }
            for f in features
        ]
        batch = base_collator(text_features)
        
        batch['trajectory_features'] = torch.stack([
            f['trajectory_features'].squeeze(0)
            for f in features
        ])
        batch['traj_positions'] = torch.stack([
            f['traj_positions'].squeeze(0)
            for f in features
        ])
        
        return batch
    
    return collator