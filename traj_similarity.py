import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

def find_and_save_most_similar_as_list(embeddings, target_trip_id, top_k, output_file, append=True):
    """
    找到与目标嵌入向量最相似的 top_k 个向量，并将结果追加或写入文件中。
    结果格式为：query_traj_id, similar_ids（similar_ids 是一个列表）。
    
    参数：
    embeddings (dict): 包含 trip_id 和嵌入向量的字典，键为 trip_id（字符串），值为 NumPy 数组。
    target_trip_id (str): 目标轨迹 ID。
    top_k (int): 返回的相似向量的数量。
    output_file (str): 保存结果的文件路径（例如 'similar_results.csv'）。
    append (bool): 是否追加到文件中（True 为追加，False 为覆盖）。
    
    返回：
    list: 按相似度降序排列的相似 trip_id 列表，不包括目标本身。
    """
    if target_trip_id not in embeddings:
        raise ValueError(f"目标 trip_id '{target_trip_id}' 不在 embeddings 中")

    # 获取目标嵌入向量
    target_embedding = embeddings[target_trip_id].reshape(1, -1)

    # 获取所有其他嵌入向量
    all_embeddings = np.array([emb for trip_id, emb in embeddings.items() if trip_id != target_trip_id])
    all_trip_ids = [trip_id for trip_id in embeddings.keys() if trip_id != target_trip_id]

    if len(all_embeddings) < top_k:
        raise ValueError(f"数据集中只有 {len(all_embeddings)} 个其他向量，无法返回 {top_k} 个相似向量")

    # 计算余弦相似度
    similarities = cosine_similarity(target_embedding, all_embeddings).flatten()

    # 按相似度降序排序并获取对应的 trip_id
    sorted_indices = np.argsort(similarities)[::-1]
    top_k_indices = sorted_indices[:top_k]

    # 提取 top_k 个相似的 trip_id
    similar_ids = [all_trip_ids[i] for i in top_k_indices]

    # 保存结果到文件
    save_results_as_list(target_trip_id, similar_ids, output_file, append=append)

    return similar_ids

def save_results_as_list(query_trip_id, similar_ids, output_file, append=True):
    """
    将查询的 trip_id 和最相似的 trip_id 列表保存到文件中。
    
    参数：
    query_trip_id (str): 目标轨迹 ID。
    similar_ids (list): 相似 trip_id 列表。
    output_file (str): 保存结果的文件路径。
    append (bool): 是否追加到文件中（True 为追加，False 为覆盖）。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开文件，使用追加或覆盖模式
    mode = 'a' if append else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 如果是覆盖模式或文件为空，写入表头
        if not append or os.path.getsize(output_file) == 0:
            writer.writerow(['query_traj_id', 'similar_ids'])
        # 写入数据行
        writer.writerow([query_trip_id, str(similar_ids)])

# 主程序
if __name__ == "__main__":
    # 加载数据
    file_path = '/home/pshao8/poi/UniTE-main/save_traj_embedding/foursquare_tky.npz'
    data = np.load(file_path, allow_pickle=True)
    trip_ids = data['trip_ids']
    embeddings_array = data['embeddings']

    # 将 trip_ids 和 embeddings 转换为字典
    embeddings = {str(trip_id): emb for trip_id, emb in zip(trip_ids, embeddings_array)}

    # 设置参数
    top_k = 3  # 找到 3 个最相似的向量
    output_file = './UniTE_h5_dataset/similar_results_list_tky.csv'

    # 清空输出文件（可选，如果需要从头开始）
    if os.path.exists(output_file):
        os.remove(output_file)

    # 为每个 trip_id 查找相似向量
    num_queried = 0
    for target_trip_id in embeddings.keys():
        try:
            find_and_save_most_similar_as_list(
                embeddings,
                target_trip_id,
                top_k,
                output_file,
                append=True
            )
            num_queried += 1
        except ValueError as e:
            print(f"错误（trip_id: {target_trip_id}）：{e}")

    # 打印查询的 trip_id 总数
    print(f"共查询了 .npz 文件中的 {num_queried} 个 trip_id")
    print(f"结果已保存到 {output_file}")

    # 验证保存的文件（可选，打印前几行）
    with open(output_file, 'r', encoding='utf-8') as f:
        print(f"\n保存的文件内容（前几行）：")
        for line in f.readlines()[:3]:  # 打印表头和前两条数据
            print(line.strip())