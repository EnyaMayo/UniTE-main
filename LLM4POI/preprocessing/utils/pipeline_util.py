import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import os.path as osp
from torchmetrics import RetrievalRecall, RetrievalNormalizedDCG, RetrievalMAP, RetrievalMRR


def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:
        for key, value in argparse_dict.items():
            if isinstance(value, torch.Tensor):
                argparse_dict[key] = value.numpy().tolist()
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        osp.join(run_args.save_path, 'checkpoint.pt')
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_step(model, data, ks=(1, 5, 10, 20)):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
            split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)

            input_data = {
                'x': row.x,
                'edge_index': row.adjs_t,
                'edge_attr': row.edge_attrs,
                'split_index': split_index,
                'delta_ts': row.edge_delta_ts,
                'delta_ss': row.edge_delta_ss,
                'edge_type': row.edge_types
            }

            out, loss = model(input_data, label=row.y[:, 0], mode='test')
            loss_list.append(loss.cpu().detach().numpy().tolist())
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")

    # Initialize torchmetrics
    recall_metrics = {k: RetrievalRecall(top_k=k) for k in ks}
    ndcg_metrics = {k: RetrievalNormalizedDCG(top_k=k) for k in ks}
    map_metrics = {k: RetrievalMAP(top_k=k) for k in ks}
    mrr_metric = RetrievalMRR()
    indexes = torch.arange(len(pred_)).unsqueeze(1)  # generate index

    for k_ in ks:
        recalls[k_] = recall_metrics[k_](pred_, label_, indexes=indexes).cpu().detach().numpy().tolist()
        logging.info(f"[Evaluating]CorrectIndex@{k_}:{torch.where(label_==pred_[:, :k_])[0].tolist()}")
        NDCGs[k_] = ndcg_metrics[k_](pred_, label_, indexes=indexes).cpu().detach().numpy().tolist()
        MAPs[k_] = map_metrics[k_](pred_, label_, indexes=indexes).cpu().detach().numpy().tolist()
        print(f'{recalls[k_]}')
        logging.info(f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = mrr_metric(pred_, label_, indexes=indexes).cpu().detach().numpy().tolist()
    logging.info(f"[Evaluating] MRR : {mrr_res}")
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)