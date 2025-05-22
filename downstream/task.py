import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import torch
from einops import repeat, rearrange
from downstream.trainer import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
import datetime

SET_NAMES = [(0, 'train'), (1, 'val'), (2, 'test')]

def next_batch(data, batch_size):
    # 批次生成器，支持元组输入
    for i in range(0, len(data[0]), batch_size):
        yield (
            data[0][i:i+batch_size],  # trips
            data[1][i:i+batch_size],  # lengths
            data[2][i:i+batch_size]   # trip_ids
        )

class Classification(Trainer):
    """ 
    A helper class for trajectory classification. 
    Class label is typically user or driver ID.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name='classification', metric_type='classification', **kwargs)
        self.loss_func = F.cross_entropy

    def cal_label(self, label_meta):
        return torch.tensor(label_meta).long().to(self.device)


class Destination(Trainer):
    """ 
    A helper class for destination prediction. 
    Feeds the encoders with truncated trajectories, 
    then regard the destinations of trajectories (last point) as prediction target.
    """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name='destination', metric_type='classification', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.cross_entropy

    def forward_encoders(self, *x):
        if len(x) < 2:
            return super().forward_encoders(*x)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:])

    def cal_label(self, label_meta):
        if label_meta[0].dim() == 2:
            return label_meta[0][:, -1].long().detach()
        return label_meta[0][:, -1, 1].long().detach()


class TTE(Trainer):
    """ 
    A helper class for travel time estimation evaluation. 
    The prediction targets is the time span (in minutes) of trajectories.
    """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name=f'tte', metric_type='regression', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.mse_loss

    def forward_encoders(self, *x):
        if len(x) < 2:
            return super().forward_encoders(*x)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:])

    def cal_label(self, label_meta):
        return torch.tensor(label_meta).float().to(self.device)


class Search(Trainer):
    """
    A helper class for similar trajectory evaluation.
    """

    def __init__(self, sim_indices=[], **kwargs):
        config = kwargs.get('config', {})
        self.triplet_indices = config.pop('triplet_indices', kwargs.pop('triplet_indices', 'ksegsimidx-100-200'))
        super().__init__(task_name='search', metric_type='classification', **kwargs)
        self.sim_indices = sim_indices
        self.models = kwargs.get('models', [])
        self.data = kwargs.get('data')
        self.device = kwargs.get('device')
        self.batch_size = kwargs.get('batch_size', 32)
        
        # 动态生成文件名
        save_dir = 'save_traj_embedding'
        os.makedirs(save_dir, exist_ok=True)

        # 获取参数
        dataset_name = getattr(self.data, 'name', 'unknown_dataset')
        epoch = kwargs.get('epoch', 'final')
        model_name = kwargs.get('model_name', 'model')
        # 时间戳
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 组合文件名
        file_name = f"{dataset_name}_{model_name}_epoch{epoch}_{now}.npz"
        self.save_path = os.path.join(save_dir, file_name)

    def train(self):
        print("Similar Trajectory Search do not require training.")
        return self.models, self.predictor

    def cal_label(self, label_meta):
        return label_meta

    def prepare_batch_meta(self, batch_meta):
        # 直接处理 (batch_trips, batch_lengths, batch_trip_ids)
        meta_prepare_func = BatchPreparer.fetch_prepare_func('trip')
        return meta_prepare_func(batch_meta, self.device)
    
    def prepare_sim_indices(self, select_set):
        dataset_name = getattr(self.data, 'name', 'unknown_dataset')
        sim_file = f'cache/meta/{dataset_name}/{self.triplet_indices}_{select_set}.npz'
        assert os.path.exists(sim_file), f"Similarity meta file not found: {sim_file}"
        loaded = np.load(sim_file, allow_pickle=True)
        qry_idx = loaded['qry_idx']
        tgt_idx = loaded['tgt_idx']
        neg_idx = loaded['neg_idx']
        return qry_idx.astype(int), tgt_idx.astype(int), neg_idx.astype(int)


    def eval(self, set_index, full_metric=True):
        print("Similar Trajectory Search do not require training.")
        
        # 1. 始终生成全量embedding（train+val+test）
        all_trips, all_lengths, all_trip_ids = [], [], []
        max_trip_len = 107
        for i in range(3):
            try:
                trips, lengths, trip_ids = self.data.load_meta('trip', i)
                if trip_ids is None:
                    raise ValueError(f"trip_ids not found in trip_{i}.npz")
                if trips.shape[1] < max_trip_len:
                    pad_len = max_trip_len - trips.shape[1]
                    pad = np.repeat(trips[:, -1:, :], pad_len, axis=1)
                    trips = np.concatenate([trips, pad], axis=1)
                all_trips.append(trips)
                all_lengths.append(lengths)
                all_trip_ids.append(trip_ids)
                print(f"Loaded trip_{i}.npz: {trips.shape[0]} trips")
            except FileNotFoundError:
                print(f"Warning: trip_{i}.npz not found, skipping.")
        
        if not all_trips:
            raise ValueError("No trip_*.npz files found for embedding generation.")
        
        trips = np.concatenate(all_trips, axis=0)
        lengths = np.concatenate(all_lengths, axis=0)
        trip_ids = np.concatenate(all_trip_ids, axis=0)
        print(f"Total trips: {trips.shape[0]}, trip_ids: {len(trip_ids)}")
        
        # 2. 生成全量embedding并保存
        self.num_iter = (len(trips) + self.batch_size - 1) // self.batch_size
        self.eval_state()
        ex_meta = self.prepare_ex_meta(None)
        
        embeds = []
        for batch_meta in tqdm(next_batch((trips, lengths, trip_ids), self.batch_size),
                               desc="Calculating embeds on full dataset",
                               total=self.num_iter,
                               leave=False):
            batch_meta = self.prepare_batch_meta(batch_meta)
            encodes = self.forward_encoders(*batch_meta, *ex_meta)
            embeds.append(encodes.detach().cpu().numpy())
        embeds = np.concatenate(embeds, 0) if embeds else np.array([])
        print(f"Generated embeddings: shape={embeds.shape}, trip_ids={len(trip_ids)}")
        
        print(f"Will save embeddings to {self.save_path}")
        np.savez(self.save_path, trip_ids=trip_ids, embeddings=embeds)
        
        # 3. 评估时只用test set的similarity meta index在全量embedding上索引
        set_name = SET_NAMES[set_index][1]
        print(f"Evaluating set_index={set_index}, set_name={set_name}")
        qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
        # 只用test set的index在全量embedding上索引
        pres, labels = self.cal_pres_and_labels(embeds[qry_idx], embeds[tgt_idx], embeds[neg_idx])
        
        if full_metric:
            self.metric_and_save(labels, pres, set_name)
        else:
            if self.metric_type == 'regression':
                mape = mean_absolute_percentage_error(labels, pres)
                return mape, 1 / (mape + 1e-6)
            elif self.metric_type == 'classification':
                acc = accuracy_score(labels, pres.argmax(-1))
                return acc, acc

    def cal_pres_and_labels(self, query, target, negs):
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[0]
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
        negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat

        labels = np.zeros(num_queries)

        return pres, labels
