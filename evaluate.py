# pip install faiss-cpu first

import numpy as np
import faiss

def read_ivecs(path):
    data = np.fromfile(path, dtype='int32')
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:]

def recall_at_k(pred, gt, k):
    if pred.shape[1] < k or gt.shape[1] < k:
        raise ValueError(f"Not enough columns to compute recall@{k}")
    pred = pred[:, :k].astype('int32')
    gt = gt[:, :k].astype('int32')
    hits = faiss.eval_intersection(pred, gt)
    return hits / (pred.shape[0] * k)

if __name__ == "__main__":
    gt_path = "siftsmall_groundtruth.ivecs"
    pred_path = "predicted.ivecs"

    gt = read_ivecs(gt_path)
    pred = read_ivecs(pred_path)

    for k in [1, 10, 100]:
        r = recall_at_k(pred, gt, k)
        print(f"Recall@{k}: {r:.4f}")
