import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def elementwise_median(param_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    n_params = len(param_lists[0])
    med = []
    for i in range(n_params):
        stacked = np.stack([p[i] for p in param_lists], axis=0)
        med.append(np.median(stacked, axis=0).astype(np.float32))
    return med

def elementwise_trimmed_mean(param_lists: List[List[np.ndarray]], trim_fraction=0.1) -> List[np.ndarray]:
    n_params = len(param_lists[0])
    trimmed = []
    n_clients = len(param_lists)
    k = int(np.floor(trim_fraction * n_clients))
    for i in range(n_params):
        stacked = np.stack([p[i] for p in param_lists], axis=0)
        sorted_vals = np.sort(stacked, axis=0)
        if k == 0:
            mean = np.mean(sorted_vals, axis=0)
        else:
            mean = np.mean(sorted_vals[k:n_clients - k], axis=0)
        trimmed.append(mean.astype(np.float32))
    return trimmed

def flatten_params(param_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.ravel() for p in param_list], axis=0)

def krum_aggregate(param_lists: List[List[np.ndarray]], f: int = 1) -> List[np.ndarray]:
    n = len(param_lists)
    if n == 0:
        raise ValueError("No client updates provided to Krum.")
    if n == 1:
        return param_lists[0]
    flattened = np.stack([flatten_params(pl) for pl in param_lists], axis=0)
    dists = np.sum((flattened[:, None, :] - flattened[None, :, :]) ** 2, axis=2)
    nb = max(1, n - f - 2)
    scores = []
    for i in range(n):
        dist_i = np.delete(dists[i], i)
        smallest = np.sort(dist_i)[:nb]
        scores.append(np.sum(smallest))
    chosen = int(np.argmin(scores))
    logger.info(f"Krum selected client index {chosen} (scores: {scores})")
    return param_lists[chosen]

class SecureAggregationStrategy(FedAvg):
    def __init__(self, secure_mode="plain", robust_agg="median", trimmed_fraction=0.1, krum_f=1, **kwargs):
        super().__init__(**kwargs)
        self.secure_mode = secure_mode
        self.robust_agg = robust_agg
        self.trimmed_fraction = trimmed_fraction
        self.krum_f = krum_f

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None

        client_params = []
        client_mask_shares = []
        mask_shares_found = True

        for _, fit_res in results:
            params = fit_res.parameters
            nd_arrays = parameters_to_ndarrays(params)
            client_params.append([np.array(a) for a in nd_arrays])
            metrics = fit_res.metrics or {}
            mask_pickled = metrics.get("mask_share")
            if mask_pickled is None:
                mask_shares_found = False
                client_mask_shares.append(None)
            else:
                try:
                    mask_share = pickle.loads(mask_pickled)
                    client_mask_shares.append([np.array(a) for a in mask_share])
                except Exception:
                    mask_shares_found = False
                    client_mask_shares.append(None)

        if self.secure_mode == "simulated_secure" and mask_shares_found:
            true_params = [[m - s for m, s in zip(masked, mask)] for masked, mask in zip(client_params, client_mask_shares)]
            candidates = true_params
        else:
            candidates = client_params

        if self.robust_agg == "median":
            agg_params = elementwise_median(candidates)
        elif self.robust_agg == "trimmed_mean":
            agg_params = elementwise_trimmed_mean(candidates, self.trimmed_fraction)
        elif self.robust_agg == "krum":
            agg_params = krum_aggregate(candidates, f=self.krum_f)
        else:
            agg_params = [np.mean(np.stack([p[i] for p in candidates], axis=0), axis=0) for i in range(len(candidates[0]))]

        return ndarrays_to_parameters(agg_params)
