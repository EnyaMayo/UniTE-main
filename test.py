# 下面这段统计yaml里的离散变量的类别数
import numpy as np
import os

# 配置
trip_files = [
    'cache/meta/nyc/trip_0.npz',
    'cache/meta/nyc/trip_1.npz',
    'cache/meta/nyc/trip_2.npz',
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


# ## 下面这段打印训练过程的loss
# # import pandas as pd

# # file_path = '/home/pshao8/poi/UniTE-main/cache/log/contrastive_b32-lr0.0001_contra-0/triplet-margin1.0-latent128/foursquare_nyc_trip/Transformer-1,2,5,6,0,3,4,7-d128-h128-l4-h8-pass/t2025_05_20_11_40_36_e0_r0.h5'
# # # 直接读取 pretrain_log
# # df = pd.read_hdf(file_path, key='pretrain_log')
# # print(df)

# # 下面这段检查npz数据类型是否合法
# # import numpy as np
# # npz = np.load('cache/meta/foursquare_nyc/trip_0.npz')
# # trips = npz['trips']
# # for idx in [1, 2, 5, 6]:  # dis_feats
# #     print(f'col {idx}: min={trips[..., idx].min()}, max={trips[..., idx].max()}, unique={np.unique(trips[..., idx]).size}')
# # print('col 1 dtype:', trips[..., 1].dtype)
# # print('col 2 dtype:', trips[..., 2].dtype)
# # print('col 5 dtype:', trips[..., 5].dtype)
# # print('col 6 dtype:', trips[..., 6].dtype)


# # 下面的这段代码统计h5长度在6-120的轨迹数量是不是和保存的npz里的一致
# import pandas as pd
# import numpy as np

# # 读取 h5 文件
# h5_path = '/home/pshao8/poi/UniTE_h5_dataset/foursquare_tky.h5'
# with pd.HDFStore(h5_path, 'r') as store:
#     trips = store['trips']

# # 统计有效轨迹
# valid_trips = [
#     trip for trip, group in trips.groupby('trip')
#     if not group.isna().any().any() and 6 <= len(group) <= 120
# ]
# print("h5中有效轨迹数:", len(valid_trips))

# # 读取 npz 文件
# embed_path = '/home/pshao8/poi/UniTE-main/save_traj_embedding/foursquare_tky_model_epochfinal_20250521_171947.npz'
# embed_data = np.load(embed_path)
# embed_trip_ids = embed_data['trip_ids']
# print("npz轨迹数:", len(embed_trip_ids))

# # 对比
# if len(valid_trips) == len(embed_trip_ids):
#     print("数量一致！")
# else:
#     print("数量不一致！")