import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import torch
from einops import repeat, rearrange
from downstream.trainer import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error

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
    # def __init__(self, sim_indices=[], **kwargs):
    #     super(Search, self).__init__(task_name=f'search', metric_type='classification', **kwargs)

    #     self.sim_indices = sim_indices

    #     # 从 kwargs 获取 dataset_name，默认为 'foursquare_nyc'
    #     dataset_name = kwargs.get('dataset_name', 'foursquare_nyc')
    #     save_dir = 'save_traj_embedding'
    #     os.makedirs(save_dir, exist_ok=True)
    #     self.save_path = os.path.join(save_dir, f'{dataset_name}.npy')
    def __init__(self, sim_indices=[], **kwargs):
        super(Search, self).__init__(task_name='search', metric_type='classification', **kwargs)
        self.sim_indices = sim_indices
        self.models = kwargs.get('models', [])
        self.data = kwargs.get('data')
        self.device = kwargs.get('device')
        self.batch_size = kwargs.get('batch_size', 32)
        
        # 设置 save_path
        # dataset_name = kwargs.get('dataset_name', 'foursquare_nyc')
        save_dir = 'save_traj_embedding'
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, f'foursquare_tky_20epoch.npz')

    def train(self):
        print("Similar Trajectory Search do not require training.")
        return self.models, self.predictor

    def cal_label(self, label_meta):
        return label_meta

    # def prepare_batch_meta(self, batch_meta):
    #     zipped = list(zip(*batch_meta))
    #     enc_meta = []
    #     for i in self.enc_meta_i:
    #         meta_prepare_func = BatchPreparer.fetch_prepare_func(self.meta_types[i])
    #         enc_meta += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
    #     return enc_meta

    # def prepare_batch_meta(self, batch_meta):
    #     # 直接处理 (batch_trips, batch_lengths, batch_trip_ids)
    #     meta_prepare_func = BatchPreparer.fetch_prepare_func('trip')
    #     return meta_prepare_func(batch_meta, self.device)

    def prepare_batch_meta(self, batch_meta):
        # 直接处理 (batch_trips, batch_lengths, batch_trip_ids)
        meta_prepare_func = BatchPreparer.fetch_prepare_func('trip')
        return meta_prepare_func(batch_meta, self.device)
    
    def prepare_sim_indices(self, select_set):
        assert len(self.sim_indices) == 1, "Only support one similarity meta now."

        sim_meta_type = self.sim_indices[0]
        qry_idx, tgt_idx, neg_idx = self.data.load_meta(sim_meta_type, select_set)

        return qry_idx.astype(int), tgt_idx.astype(int), neg_idx.astype(int)

    # def eval(self, set_index, full_metric=True):
    #     set_name = SET_NAMES[set_index][1]
    #     self.prepare_batch_iter(set_index)
    #     self.eval_state()

    #     ex_meta = self.prepare_ex_meta(set_index)

    #     embeds = []
    #     for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
    #                            desc=f"Calculating embeds on {set_name} set",
    #                            total=self.num_iter, leave=False):
    #         batch_meta = self.prepare_batch_meta(batch_meta)
    #         encodes = self.forward_encoders(*batch_meta, *ex_meta)
    #         embeds.append(encodes.detach().cpu().numpy())
    #     embeds = np.concatenate(embeds, 0)
    #     ## shaoyun
    #     np.save(self.save_path, embeds)
    #     print(f'Trajectory embeddings saved to {self.save_path}')
    #     ## end
        
    #     qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
    #     pres, labels = self.cal_pres_and_labels(embeds[qry_idx], embeds[tgt_idx], embeds[neg_idx])

    #     if full_metric:
    #         self.metric_and_save(labels, pres, set_name)
    #     else:
    #         if self.metric_type == 'regression':
    #             mape = mean_absolute_percentage_error(labels, pres)
    #             return mape, 1 / (mape + 1e-6)
    #         elif self.metric_type == 'classification':
    #             acc = accuracy_score(labels, pres.argmax(-1))
    #             return acc, acc

    # def eval(self, set_index, full_metric=True):
    #     print("Similar Trajectory Search do not require training.")
        
    #     try:
    #         # 加载所有 trip_*.npz
    #         all_trips, all_lengths, all_trip_ids = [], [], []
    #         max_trip_len = 107
    #         for i in range(3):
    #             try:
    #                 trips, lengths, trip_ids = self.data.load_meta('trip', i)
    #                 if trip_ids is None:
    #                     raise ValueError(f"trip_ids not found in trip_{i}.npz")
    #                 # 填充 trips 到 max_trip_len
    #                 if trips.shape[1] < max_trip_len:
    #                     pad_len = max_trip_len - trips.shape[1]
    #                     pad = np.repeat(trips[:, -1:, :], pad_len, axis=1)
    #                     trips = np.concatenate([trips, pad], axis=1)
    #                 all_trips.append(trips)
    #                 all_lengths.append(lengths)
    #                 all_trip_ids.append(trip_ids)
    #             except FileNotFoundError:
    #                 print(f"Warning: trip_{i}.npz not found, skipping.")
    #         if not all_trips:
    #             raise ValueError("No trip_*.npz files found.")
    #         trips = np.concatenate(all_trips, axis=0)
    #         lengths = np.concatenate(all_lengths, axis=0)
    #         trip_ids = np.concatenate(all_trip_ids, axis=0)
            
    #         self.prepare_batch_iter(None)
    #         self.eval_state()
    #         ex_meta = self.prepare_ex_meta(None)
            
    #         embeds = []
    #         for batch_meta in tqdm(next_batch([], self.batch_size),
    #                                desc="Calculating embeds on full dataset",
    #                                total=self.num_iter, leave=False):
    #             batch_meta = self.prepare_batch_meta(batch_meta)
    #             encodes = self.forward_encoders(*batch_meta, *ex_meta)
    #             embeds.append(encodes.detach().cpu().numpy())
    #         embeds = np.concatenate(embeds, 0) if embeds else np.array([])
    #         print(f"Loaded trips: {embeds.shape[0]} trips, embed shape: {embeds.shape}, trip_ids: {len(trip_ids)}")
            
    #         print(f"Saving embeddings to {self.save_path}, shape: {embeds.shape}, trip_ids: {len(trip_ids)}")
    #         np.savez(self.save_path, trip_ids=trip_ids, embeddings=embeds)
    #     except FileNotFoundError:
    #         raise ValueError("No trip_*.npz files found.")
        
    #     set_name = SET_NAMES[set_index][1]
    #     self.prepare_batch_iter(set_index)
    #     self.eval_state()
    #     ex_meta = self.prepare_ex_meta(set_index)
        
    #     embeds = []
    #     for batch_meta in tqdm(next_batch([], self.batch_size),
    #                            desc=f"Calculating embeds on {set_name} set",
    #                            total=self.num_iter, leave=False):
    #         batch_meta = self.prepare_batch_meta(batch_meta)
    #         encodes = self.forward_encoders(*batch_meta, *ex_meta)
    #         embeds.append(encodes.detach().cpu().numpy())
    #     embeds = np.concatenate(embeds, 0) if embeds else np.array([])
        
    #     qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
    #     pres, labels = self.cal_pres_and_labels(embeds[qry_idx], embeds[tgt_idx], embeds[neg_idx])
        
    #     if full_metric:
    #         self.metric_and_save(labels, pres, set_name)
    #     else:
    #         if self.metric_type == 'regression':
    #             mape = mean_absolute_percentage_error(labels, pres)
    #             return mape, 1 / (mape + 1e-6)
    #         elif self.metric_type == 'classification':
    #             acc = accuracy_score(labels, pres.argmax(-1))
    #             return acc, acc
    # def eval(self, set_index, full_metric=True):
    #     print("Similar Trajectory Search do not require training.")
        
    #     # 加载所有 trip_*.npz
    #     all_trips, all_lengths, all_trip_ids = [], [], []
    #     max_trip_len = 107
    #     for i in range(3):
    #         try:
    #             trips, lengths, trip_ids = self.data.load_meta('trip', i)
    #             if trip_ids is None:
    #                 raise ValueError(f"trip_ids not found in trip_{i}.npz")
    #             if trips.shape[1] < max_trip_len:
    #                 pad_len = max_trip_len - trips.shape[1]
    #                 pad = np.repeat(trips[:, -1:, :], pad_len, axis=1)
    #                 trips = np.concatenate([trips, pad], axis=1)
    #             all_trips.append(trips)
    #             all_lengths.append(lengths)
    #             all_trip_ids.append(trip_ids)
    #             print(f"Loaded trip_{i}.npz: {trips.shape[0]} trips")
    #         except FileNotFoundError:
    #             print(f"Warning: trip_{i}.npz not found, skipping.")
        
    #     if not all_trips:
    #         raise ValueError("No trip_*.npz files found for embedding generation.")
        
    #     trips = np.concatenate(all_trips, axis=0)
    #     lengths = np.concatenate(all_lengths, axis=0)
    #     trip_ids = np.concatenate(all_trip_ids, axis=0)
    #     print(f"Total trips: {trips.shape[0]}, trip_ids: {len(trip_ids)}")
        
    #     # 手动构造批次迭代器
    #     self.num_iter = (len(trips) + self.batch_size - 1) // self.batch_size
    #     self.eval_state()
    #     ex_meta = self.prepare_ex_meta(None)
        
    #     embeds = []
    #     for batch_meta in tqdm(next_batch((trips, lengths, trip_ids), self.batch_size),
    #                            desc="Calculating embeds on full dataset",
    #                            total=self.num_iter,
    #                            leave=False):
    #         batch_meta = self.prepare_batch_meta(batch_meta)
    #         encodes = self.forward_encoders(*batch_meta, *ex_meta)
    #         embeds.append(encodes.detach().cpu().numpy())
    #     embeds = np.concatenate(embeds, 0) if embeds else np.array([])
    #     print(f"Generated embeddings: shape={embeds.shape}, trip_ids={len(trip_ids)}")
        
    #     print(f"Saving embeddings to {self.save_path}, shape={embeds.shape}, trip_ids={len(trip_ids)}")
    #     np.savez(self.save_path, trip_ids=trip_ids, embeddings=embeds)
        
    #     # 评估特定集合
    #     set_name = SET_NAMES[set_index][1]
    #     batch_iter = self.prepare_batch_iter(set_index)
    #     self.num_iter = len(batch_iter) // self.batch_size if batch_iter else 0
    #     self.eval_state()
    #     ex_meta = self.prepare_ex_meta(set_index)
        
    #     embeds = []
    #     for batch_meta in tqdm(batch_iter,
    #                            desc=f"Calculating embeds on {set_name} set",
    #                            total=self.num_iter,
    #                            leave=False):
    #         batch_meta = self.prepare_batch_meta(batch_meta)
    #         encodes = self.forward_encoders(*batch_meta, *ex_meta)
    #         embeds.append(encodes.detach().cpu().numpy())
    #     embeds = np.concatenate(embeds, 0) if embeds else np.array([])
        
    #     qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
    #     pres, labels = self.cal_pres_and_labels(embeds[qry_idx], embeds[tgt_idx], embeds[neg_idx])
        
    #     if full_metric:
    #         self.metric_and_save(labels, pres, set_name)
    #     else:
    #         if self.metric_type == 'regression':
    #             mape = mean_absolute_percentage_error(labels, pres)
    #             return mape, 1 / (mape + 1e-6)
    #         elif self.metric_type == 'classification':
    #             acc = accuracy_score(labels, pres.argmax(-1))
    #             return acc, acc

    def eval(self, set_index, full_metric=True):
        print("Similar Trajectory Search do not require training.")
        
        # 加载所有 trip_*.npz
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
        
        # 手动构造批次迭代器
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
        
        print(f"Saving embeddings to {self.save_path}, shape={embeds.shape}, trip_ids={len(trip_ids)}")
        np.savez(self.save_path, trip_ids=trip_ids, embeddings=embeds)
        
        # 评估特定集合
        set_name = SET_NAMES[set_index][1]
        print(f"Evaluating set_index={set_index}, set_name={set_name}")
        batch_iter = self.prepare_batch_iter(set_index)
        
        # 调试 batch_iter
        if batch_iter is None:
            print(f"Error: batch_iter is None for {set_name} set.")
        elif not batch_iter:
            print(f"Error: batch_iter is empty for {set_name} set.")
        else:
            print(f"batch_iter length: {len(batch_iter)}")
        
        # 如果 batch_iter 无效，尝试手动加载
        if batch_iter is None or not batch_iter:
            print(f"Warning: Invalid batch_iter for {set_name} set, attempting manual load.")
            try:
                trips, lengths, trip_ids = self.data.load_meta('trip', set_index)
                if trips.size == 0 or lengths.size == 0 or trip_ids.size == 0:
                    print(f"Error: Empty data in trip_{set_index}.npz, skipping evaluation.")
                    return 0, 0
                print(f"Manually loaded trip_{set_index}.npz: {trips.shape[0]} trips")
                batch_iter = list(next_batch((trips, lengths, trip_ids), self.batch_size))
                if not batch_iter:
                    print(f"Error: No batches generated for trip_{set_index}.npz, skipping evaluation.")
                    return 0, 0
            except FileNotFoundError:
                print(f"Error: trip_{set_index}.npz not found, skipping evaluation.")
                return 0, 0
        
        self.num_iter = (len(batch_iter) + self.batch_size - 1) // self.batch_size
        self.eval_state()
        ex_meta = self.prepare_ex_meta(set_index)
        
        embeds = []
        for batch_meta in tqdm(batch_iter,
                               desc=f"Calculating embeds on {set_name} set",
                               total=self.num_iter,
                               leave=False):
            batch_meta = self.prepare_batch_meta(batch_meta)
            encodes = self.forward_encoders(*batch_meta, *ex_meta)
            embeds.append(encodes.detach().cpu().numpy())
        embeds = np.concatenate(embeds, 0) if embeds else np.array([])
        
        if embeds.size == 0:
            print(f"Warning: No embeddings generated for {set_name} set, skipping metrics.")
            return 0, 0
        
        qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
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
