import json
import os
from collections import defaultdict

def process_files(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 收集所有poi_id
    poi_ids = set()
    
    # 首先遍历所有文件收集所有唯一的poi_id
    for filename in ['test.json', 'train.json', 'val.json']:
        input_path = os.path.join(input_folder, filename)
        if not os.path.exists(input_path):
            continue
        
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    poi_ids.add(data['poi_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # 创建poi_id到label的映射（从0开始）
    poi_to_label = {poi_id: idx for idx, poi_id in enumerate(sorted(poi_ids))}
    
    # 处理每个文件
    for filename in ['test.json', 'train.json', 'val.json']:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_label.json")
        
        if not os.path.exists(input_path):
            continue
        
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                try:
                    data = json.loads(line.strip())
                    # 创建新数据，添加label字段
                    new_data = {
                        'trajectory_features': data['trajectory_features'],
                        'poi_id': data['poi_id'],
                        'label': poi_to_label[data['poi_id']]
                    }
                    f_out.write(json.dumps(new_data) + '\n')
                except (json.JSONDecodeError, KeyError):
                    continue
    num_unique_poi_ids = len(poi_ids)
    print(f"唯一 poi_id 的数量: {num_unique_poi_ids}")  # 打印到控制台

if __name__ == "__main__":
    input_folder = "./foursquare_nyc"  # 输入文件夹路径，假设脚本与JSON文件在同一目录
    output_folder = "./foursquare_nyc"  # 输出文件夹路径
    
    process_files(input_folder, output_folder)
    print("处理完成，结果已保存到", output_folder)


#2880 poi for nyc