import pickle
from pathlib import Path
import glob
import re
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp



def load_data(data_file):
    dataset_dir = "dataset/"
     
    # Load the preprocessed data from the pickle file
    with open( dataset_dir +data_file, "rb") as file:
        X_train, X_Val, X_test = pickle.load(file)
    return X_train, X_Val, X_test

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def load_graph_adj_mtx(pkl_file):
    mapfolder = "dataset/tfm/"
    with open(mapfolder+pkl_file, 'rb') as f:
        graph_dict = pickle.load(f)
    graph_data = nx.DiGraph(graph_dict)
    adj_mtx = nx.adjacency_matrix(graph_data)
    return adj_mtx

def load_graph(pkl_file):
    mapfolder = "dataset/tfm/"
    with open(mapfolder+pkl_file, 'rb') as f:
        graph_dict = pickle.load(f)
    graph_data = nx.DiGraph(graph_dict)
    return graph_data

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # Convert adjacency matrix to sparse matrix
    adj_mat = sp.csr_matrix(adj_mat)

    # Compute the degree matrix
    deg_mat = sp.diags(adj_mat.sum(axis=1).A1)

    # Identity matrix
    id_mat = sp.eye(n_vertex)

    if mat_type == 'com_lap_mat':
        # Combinatorial Laplacian matrix
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # Random walk normalized Laplacian matrix for ChebConv
        deg_mat_inv = sp.diags(1 / deg_mat.diagonal())
        rw_lap_mat = deg_mat_inv @ adj_mat
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # Random walk normalized Laplacian matrix for GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        wid_deg_mat_inv = sp.diags(1 / wid_deg_mat.diagonal())
        hat_rw_normd_lap_mat = wid_deg_mat_inv @ wid_adj_mat
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss