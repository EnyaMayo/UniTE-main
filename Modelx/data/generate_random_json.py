import json
import numpy as np

def generate_sample(poi_id):
    """生成一个样本，包含4条1024维的轨迹特征"""
    return {
        "trajectory_features": [
            np.random.uniform(0, 1, 1024).round(2).tolist(),
            np.random.uniform(0, 1, 1024).round(2).tolist(),
            np.random.uniform(0, 1, 1024).round(2).tolist(),
            np.random.uniform(0, 1, 1024).round(2).tolist()
        ],
        "poi_id": poi_id
    }

# 生成train.json
with open("train.json", "w") as f:
    for poi_id in [42, 43]:
        json.dump(generate_sample(poi_id), f)
        f.write("\n")

# 生成test.json
with open("test.json", "w") as f:
    for poi_id in [44, 45]:
        json.dump(generate_sample(poi_id), f)
        f.write("\n")

# 生成validation.json
with open("validation.json", "w") as f:
    for poi_id in [46, 47]:
        json.dump(generate_sample(poi_id), f)
        f.write("\n")