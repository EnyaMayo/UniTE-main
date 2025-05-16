import pandas as pd
import torch
from datetime import datetime
from collections import Counter
import numpy as np
import random
import math
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) < repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask

class ContinuousEncoding(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode

class TrajConvEmbedding(nn.Module):
    def __init__(self, d_model, dis_feats=[], num_embeds=[], con_feats=[], kernel_size=3,
                 pre_embed=None, pre_embed_update=False, second_col=None):
        super().__init__()

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col

        # Operates discrete features by look-up table.
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if pre_embed is not None:
            self.dis_embeds[0].weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                     requires_grad=pre_embed_update)

        # Operates continuous features by convolution.
        self.conv = nn.Conv1d(len(con_feats), d_model,
                              kernel_size=kernel_size, padding=(kernel_size - 1)//2)

        # Time embedding
        if second_col is not None:
            self.time_embed = ContinuousEncoding(d_model)

    def forward(self, x):
        B, L, E_in = x.shape

        h = torch.zeros(B, L, self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())
        if self.con_feats is not None:
            h += self.conv(x[..., self.con_feats].transpose(1, 2)).transpose(1, 2)

        if self.second_col is not None:
            h += self.time_embed(x[..., int(self.second_col)])

        return h

def load_txt_data(txt_path):
    """加载 TSMC2014 txt 数据"""
    columns = ['user_id', 'venue_id', 'venue_category_id', 'venue_category_name', 
               'lat', 'lng', 'Timezone_Offset', 'UTC_Time']
    df = pd.read_csv(txt_path, sep='\t', header=None, names=columns, encoding='ISO-8859-1')
    return df

def prepare_trajectory_data(user_data):
    """为轨迹嵌入准备数据"""
    # 转换为本地时间并生成 trip ID
    user_data['time'] = pd.to_datetime(user_data['UTC_Time'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    user_data['local_time'] = user_data['time'] + pd.to_timedelta(user_data['Timezone_Offset'], unit='m')
    user_data['date'] = user_data['local_time'].dt.date
    user_data['trip'] = user_data['user_id'].astype(str) + '_' + user_data['date'].astype(str)
    user_data['weekday'] = user_data['local_time'].dt.weekday
    user_data['hour'] = user_data['local_time'].dt.hour

    # 按 trip 分组
    trips = user_data.groupby('trip')
    traj_data = []
    o_pois = []
    d_pois = []
    valid_lens = []
    start_weekdays = []
    start_hours = []

    for trip_id, trip_data in trips:
        lat = torch.tensor(trip_data['lat'].values, dtype=torch.float)
        lng = torch.tensor(trip_data['lng'].values, dtype=torch.float)
        hour = torch.tensor(trip_data['hour'].values, dtype=torch.float)
        x = torch.stack([lat, lng, hour], dim=-1)  # [L, 3]
        valid_len = torch.tensor(len(trip_data), dtype=torch.long)

        o_poi = trip_data['venue_category_name'].iloc[0]
        d_poi = trip_data['venue_category_name'].iloc[-1]
        start_weekday = trip_data['weekday'].iloc[0]
        start_hour = trip_data['hour'].iloc[0]

        traj_data.append(x)
        o_pois.append(o_poi)
        d_pois.append(d_poi)
        valid_lens.append(valid_len)
        start_weekdays.append(start_weekday)
        start_hours.append(start_hour)

    max_len = max(len(t) for t in traj_data)
    traj_data = [torch.cat([t, torch.zeros(max_len - len(t), 3)], dim=0) for t in traj_data]
    traj_data = torch.stack(traj_data, dim=0)  # [B, L, 3]
    valid_lens = torch.stack(valid_lens, dim=0)  # [B]
    start_weekdays = torch.tensor(start_weekdays, dtype=torch.long)
    start_hours = torch.tensor(start_hours, dtype=torch.long)

    return traj_data, valid_lens, o_pois, d_pois, start_weekdays, start_hours

def extract_user_patterns(df, model, tokenizer, embedder, device='cuda'):
    """生成用户描述和分类"""
    df = df.dropna(subset=['local_time', 'lat', 'lng', 'venue_category_name'])
    
    # 打印 top 30 POI
    poi_counts = Counter(df['venue_category_name'])
    total_checkins = sum(poi_counts.values())
    top_pois = pd.Series(poi_counts).sort_values(ascending=False)[:30]
    print("\nTop 30 POI 类别（百分比）：")
    print((top_pois / total_checkins * 100).to_string())

    user_groups = df.groupby('user_id')
    user_descriptions = {}
    user_classifications = {}

    for user_id, user_data in user_groups:
        # 时间模式
        local_times = pd.to_datetime(user_data['local_time'])
        weekdays = local_times.dt.weekday
        hours = local_times.dt.hour
        time_patterns = []
        if (weekdays < 5).mean() > 0.8 and ((hours >= 7) & (hours <= 9)).mean() > 0.2:
            time_patterns.append("regular commuting")
        if ((hours >= 11) & (hours <= 14)).mean() > 0.15 or ((hours >= 18) & (hours <= 20)).mean() > 0.15:
            time_patterns.append("frequent dining")
        if ((hours >= 18) & (hours <= 22)).mean() > 0.15:
            time_patterns.append("active socializing")
        if (hours >= 20).mean() > 0.15:
            time_patterns.append("regular nightlife")
        if (weekdays >= 5).mean() > 0.3:
            time_patterns.append("leisurely weekend activities")

        # POI 偏好
        poi_counts = Counter(user_data['venue_category_name'])
        total_checkins = sum(poi_counts.values())
        top_pois = [category for category, count in poi_counts.most_common(3) if count / total_checkins > 0.15]
        poi_description = f"frequently visits {', '.join(top_pois)}" if top_pois else "diverse locations"

        # 轨迹长度和频率
        trip_lengths = user_data.groupby('trip').size()
        avg_length = trip_lengths.mean()
        lifestyle = "urban destinations" if avg_length > 5 else "local destinations"

        # 出行偏好
        lat_std = user_data['lat'].std()
        lng_std = user_data['lng'].std()
        travel_pref = "long-distance traveler" if (lat_std > 0.1 or lng_std > 0.1) else "local explorer"

        # 描述
        time_pattern_str = ', '.join(time_patterns) if time_patterns else "varied patterns"
        description = f"User {user_id} exhibits {time_pattern_str}. They {poi_description}, prefer {travel_pref}, and have {lifestyle}."
        user_descriptions[user_id] = description

        # 准备轨迹数据
        traj_data, valid_lens, o_pois, d_pois, start_weekdays, start_hours = prepare_trajectory_data(user_data)
        traj_data = traj_data.to(device)
        valid_lens = valid_lens.to(device)
        start_weekdays = start_weekdays.to(device)
        start_hours = start_hours.to(device)

        # 轨迹嵌入（仅用于验证数据格式）
        with torch.no_grad():
            traj_emb = embedder(traj_data)  # [B, L, d_model]
            batch_mask = get_batch_mask(traj_data.shape[0], traj_data.shape[1], valid_lens).unsqueeze(-1)
            traj_emb = traj_emb * batch_mask  # 掩码填充

        # 提示分类
        prompt = f"""
        Based on the following user travel description, classify the user into one or more categories from the following: 
        Professions: Student, Programmer, Office Worker, Retail Worker, Freelancer, Service Worker
        Activities: Commuting, Dining, Socializing, Shopping, Nightlife, Tech Shopping
        Lifestyles: Urban, Social, Work-Centric, Leisurely, Minimalist
        Travel Preferences: Regular Commuter, Local Explorer, Long-Distance Traveler, Leisurely Traveler
        
        Description: {description}
        
        Output the categories that best describe the user in the format:
        Professions: ...
        Activities: ...
        Lifestyles: ...
        Travel Preferences: ...
        """
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 解析输出
        classifications = {'Professions': [], 'Activities': [], 'Lifestyles': [], 'Travel Preferences': []}
        lines = output_text.split('\n')
        current_category = None
        for line in lines:
            line = line.strip()
            if line.startswith('Professions:'):
                current_category = 'Professions'
                classifications[current_category] = [c.strip() for c in line[len('Professions:') + 1:].split(',') if c.strip()]
            elif line.startswith('Activities:'):
                current_category = 'Activities'
                classifications[current_category] = [c.strip() for c in line[len('Activities:') + 1:].split(',') if c.strip()]
            elif line.startswith('Lifestyles:'):
                current_category = 'Lifestyles'
                classifications[current_category] = [c.strip() for c in line[len('Lifestyles:') + 1:].split(',') if c.strip()]
            elif line.startswith('Travel Preferences:'):
                current_category = 'Travel Preferences'
                classifications[current_category] = [c.strip() for c in line[len('Travel Preferences:') + 1:].split(',') if c.strip()]
            elif current_category and line:
                classifications[current_category].extend([c.strip() for c in line.split(',') if c.strip()])

        user_classifications[user_id] = classifications

    return user_descriptions, user_classifications

def main(txt_path, output_path, classification_path, model_path='/home/pshao8/poi/LLM4POI/weights/llama2/models--Yukang--Llama-2-7b-longlora-32k-ft/snapshots/ab48674ffc55568ffe2a1207ef0e711c2febbaaf', device='cuda'):
    """主函数：加载 Llama-2，提取描述，分类，生成提示"""
    # 设置种子
    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32768, padding_side="right", use_fast=True)

    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True, output_attentions=True, _flash_attn_2_enabled=True)
    context_size = 32768
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='balanced',
        config=config,
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

    # 应用 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.eval()
    model.to(device)

    # 初始化轨迹嵌入器
    embedder = TrajConvEmbedding(
        d_model=768,  # 修复：使用 d_model 而非 emb_size
        dis_feats=[0, 1],  # 纬度、经度
        num_embeds=[10, 10],
        con_feats=[2],  # 小时
        kernel_size=3
    ).to(device)

    # 加载数据
    df = load_txt_data(txt_path)

    # 提取模式和分类
    user_descriptions, user_classifications = extract_user_patterns(df, model, tokenizer, embedder, device)

    # 生成提示
    prompt_template = """
    Based on the following user travel description, classify the user into one or more categories from the following: 
    Professions: Student, Programmer, Office Worker, Retail Worker, Freelancer, Service Worker
    Activities: Commuting, Dining, Socializing, Shopping, Nightlife, Tech Shopping
    Lifestyles: Urban, Social, Work-Centric, Leisurely, Minimalist
    Travel Preferences: Regular Commuter, Local Explorer, Long-Distance Traveler, Leisurely Traveler
    
    Description: {description}
    
    Output the categories that best describe the user in the format:
    Professions: ...
    Activities: ...
    Lifestyles: ...
    Travel Preferences: ...
    """
    prompts = {user_id: prompt_template.format(description=desc) for user_id, desc in user_descriptions.items()}

    # 保存提示
    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id, prompt in prompts.items():
            f.write(f"User {user_id}:\n{prompt}\n\n")

    # 保存分类结果
    with open(classification_path, 'w', encoding='utf-8') as f:
        for user_id, classifications in user_classifications.items():
            f.write(f"User {user_id}:\n")
            for category_type, categories in classifications.items():
                f.write(f"{category_type}: {', '.join(categories) if categories else 'None'}\n")
            f.write("\n")

    # 验证输出
    print("\nSample User Descriptions and Classifications:")
    for user_id in list(user_descriptions.keys())[:3]:
        print(f"User {user_id}:")
        print(f"Description: {user_descriptions[user_id]}")
        print("Classifications:")
        for category_type, categories in user_classifications[user_id].items():
            print(f"  {category_type}: {', '.join(categories) if categories else 'None'}")
        print()

if __name__ == "__main__":
    txt_path = "/home/pshao8/poi/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"
    output_path = "./user_prompts.txt"
    classification_path = "./user_classifications.txt"
    main(txt_path, output_path, classification_path)