import numpy as np
import os

# 配置
trip_files = [
    'cache/meta/foursquare_nyc/trip_0.npz',
    'cache/meta/foursquare_nyc/trip_1.npz',
    'cache/meta/foursquare_nyc/trip_2.npz',
]
dis_feats = [1, 2, 5, 6]  # 按需修改

max_vals = {idx: 0 for idx in dis_feats}

for file in trip_files:
    if not os.path.exists(file):
        print(f'File not found: {file}')
        continue
    arr = np.load(file)['trips']
    for idx in dis_feats:
        max_val = arr[..., idx].max()
        print(f'{file} dis_feat {idx} max: {max_val}')
        if max_val > max_vals[idx]:
            max_vals[idx] = max_val

print('\n===== 建议的 num_embeds 配置 =====')
print('num_embeds:', [int(max_vals[idx]) + 1 for idx in dis_feats])
for idx in dis_feats:
    print(f'dis_feat {idx}: max={int(max_vals[idx])}, 建议num_embeds={int(max_vals[idx])+1}') 