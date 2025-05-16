import torch
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from argparse import ArgumentParser
import numpy as np
import random
import math
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import os
from scipy.stats import mode

torch.cuda.empty_cache()

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_config():
    parser = ArgumentParser(description='arg parser')
    parser.add_argument('--output_dir', type=str, default="/home/pshao8/poi/UniTE_h5_dataset")
    parser.add_argument('--context_size', type=int, default=32768)
    parser.add_argument('--seq_len', type=int, default=32768)
    args = parser.parse_args()
    return args

class DescriptionDataset(Dataset):
    def __init__(self, trip_ids, descriptions):
        self.trip_ids = trip_ids
        self.descriptions = descriptions
    
    def __len__(self):
        return len(self.trip_ids)
    
    def __getitem__(self, idx):
        return self.trip_ids[idx], self.descriptions[idx]

def compute_fea():
    hidden_states_traj = {}  # Map trip_id to hidden_states
    output = 'tky_trajectory_descriptions'
    
    # Read CSV
    csv_file = f'{data_path}{output}.csv'
    try:
        data_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file {csv_file} not found.")
        return
    
    # Pre-truncate sequences to max_length=1000
    data_df['description'] = data_df['description'].apply(
        lambda x: tokenizer.decode(tokenizer(x, max_length=1000, truncation=True)['input_ids'])
    )
    
    # Calculate sequence lengths
    lengths = [len(tokenizer(desc, max_length=None, truncation=False)['input_ids']) for desc in data_df['description']]
    lengths_np = np.array(lengths)
    
    # Compute statistics
    mean_length = np.mean(lengths_np)
    median_length = np.median(lengths_np)
    mode_length = float(mode(lengths_np, keepdims=False)[0])
    q1, q2, q3 = np.percentile(lengths_np, [25, 50, 75])
    min_length = np.min(lengths_np)
    max_length = np.max(lengths_np)
    
    # Print statistics
    print(f"Sequence length statistics (after truncation to max_length=1000):")
    print(f"  Average: {mean_length:.2f}")
    print(f"  Median: {median_length:.2f}")
    print(f"  Mode: {mode_length:.2f}")
    print(f"  Q1 (25th percentile): {q1:.2f}")
    print(f"  Q2 (50th percentile): {q2:.2f}")
    print(f"  Q3 (75th percentile): {q3:.2f}")
    print(f"  Min: {min_length:.2f}")
    print(f"  Max: {max_length:.2f}")
    
    # Group by sequence length for dynamic batching
    data_df['seq_length'] = lengths
    data_df = data_df.sort_values('seq_length', ascending=False)
    
    # Split into groups: long (>500), medium (250-500), short (<250)
    long_df = data_df[data_df['seq_length'] > 500]
    medium_df = data_df[(data_df['seq_length'] >= 250) & (data_df['seq_length'] <= 500)]
    short_df = data_df[data_df['seq_length'] < 250]
    
    # Batch sizes for each group
    batch_sizes = {
        'long': 1,    # Long sequences: process one at a time
        'medium': 2,  # Medium: batch of 2
        'short': 4    # Short: batch of 4
    }
    
    # Flag for first sample
    first_sample = True
    
    # Process each group
    for group_name, df, batch_size in [
        ('long', long_df, batch_sizes['long']),
        ('medium', medium_df, batch_sizes['medium']),
        ('short', short_df, batch_sizes['short'])
    ]:
        if len(df) == 0:
            continue
        
        print(f"Processing {group_name} sequences (batch_size={batch_size}, {len(df)} samples)")
        dataset = DescriptionDataset(df['trip_id'].tolist(), df['description'].tolist())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        for batch in tqdm(dataloader, desc=f"Processing {group_name} data"):
            try:
                trip_ids, descriptions = batch
                
                # Tokenize batch without further truncation
                inputs = tokenizer(descriptions, return_tensors="pt", padding=True, max_length=1000, truncation=True).to(device)
                with torch.no_grad(), autocast():
                    outputs = model(**inputs)
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, 4096]
                
                # Print first sample's first 20 elements
                if first_sample:
                    print(f"First trip_id: {trip_ids[0]}")
                    print(f"First 20 elements of query.hidden_states[-1][0, 0, :20]:")
                    print(hidden_states[0, 0, :20].cpu().numpy())
                    first_sample = False
                
                # Mean pooling
                hidden_states = hidden_states.mean(dim=1)  # [batch_size, 4096]
                hidden_states = hidden_states.cpu().detach()
                
                # Store hidden states
                for trip_id, hidden_state in zip(trip_ids, hidden_states):
                    hidden_states_traj[trip_id] = hidden_state.unsqueeze(0)  # [1, 4096]
                
                # Monitor GPU memory
                for gpu_id in [0, 1]:
                    torch.cuda.synchronize(f'cuda:{gpu_id}')
                    allocated = torch.cuda.memory_allocated(f'cuda:{gpu_id}') / 1024**3
                    reserved = torch.cuda.memory_reserved(f'cuda:{gpu_id}') / 1024**3
                    print(f"GPU {gpu_id} - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")
                
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
                for gpu_id in [0, 1]:
                    torch.cuda.synchronize(f'cuda:{gpu_id}')
                    print(f"GPU {gpu_id} - After cache clear - Allocated: {torch.cuda.memory_allocated(f'cuda:{gpu_id}') / 1024**3:.2f} GiB")
                
            except Exception as ex:
                if "CUDA out of memory" in str(ex):
                    torch.cuda.empty_cache()
                    for gpu_id in [0, 1]:
                        torch.cuda.synchronize(f'cuda:{gpu_id}')
                        allocated = torch.cuda.memory_allocated(f'cuda:{gpu_id}') / 1024**3
                        reserved = torch.cuda.memory_reserved(f'cuda:{gpu_id}') / 1024**3
                        print(f"GPU {gpu_id} - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")
                print(f"Error processing batch: {ex}")
                continue
    
    # Save to pickle
    output_file = f'{data_path}{output}_embedding.pkl'
    with open(output_file, 'wb') as fp:
        pkl.dump(hidden_states_traj, fp)
    print(f"Saved hidden states to {output_file}")

def main(args):
    global data_path, model, tokenizer, device
    
    # Setup device and seeds
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Model path
    model_path = '/home/pshao8/poi/LLM4POI/weights/llama2/models--Yukang--Llama-2-7b-longlora-32k-ft/snapshots/ab48674ffc55568ffe2a1207ef0e711c2febbaaf'
    data_path = f'/home/pshao8/poi/UniTE_h5_dataset/'
    print("data path", data_path)
    print("base model", model_path)
    print("peft model", args.output_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=32768,
        padding_side="right",
        use_fast=True,
    )
    
    # Load model config
    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
        output_hidden_states=True,
        output_attentions=True,
        _flash_attn_2_enabled=True
    )
    
    # Set RoPE scaling
    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='balanced',  # Distribute across GPUs
        config=config,
        cache_dir=None,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.resize_token_embeddings(32001)
    
    # Apply LoRA (no freezing)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.eval()
    
    # Run compute_fea
    compute_fea()

if __name__ == "__main__":
    args = parse_config()
    main(args)