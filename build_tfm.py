""" Build the user-agnostic global trajectory flow map from the sequence data """
""" Modify from GETNext  https://github.com/songyangme/GETNext"""

import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_data

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users:
        users.remove(exclude_user)
    print(f"Number of unique users: {len(users)}")
    
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]
        user_df = user_df.sort_values('trajectory_id')  # Sort by trajectory_id

        # Add nodes (POI)
        for i, row in user_df.iterrows():
            node = row['poi_id']
            if node not in G.nodes():
                G.add_node(row['poi_id'],
                           checkin_cnt=1,
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        poi_ids = user_df['poi_id'].tolist()
        for i in range(len(poi_ids) - 1):
            prev_poi_id = poi_ids[i]
            curr_poi_id = poi_ids[i + 1]
            
            if G.has_edge(prev_poi_id, curr_poi_id):
                G.edges[prev_poi_id, curr_poi_id]['weight'] += 1
            else:
                G.add_edge(prev_poi_id, curr_poi_id, weight=1)

    return G
def save_graph_to_pickle(G, dst_dir, dataset):
    pickle.dump(G, open(os.path.join(dst_dir, dataset+'_graph.pkl'), 'wb'))
    
def save_graph_edgelist(G, dst_dir, dataset):
    nodelist = list(G.nodes())
    node_id2idx = {str(k): v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, dataset + '_graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, dataset + '_graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            if src_node in node_id2idx and dst_node in node_id2idx:
                print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)
            else:
                print(f"Warning: Node '{src_node}' or '{dst_node}' not found in node_id2idx")

def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


def build_graph(dataset, dst_dir): 
    # Load the preprocessed data
    X_train, _, _ = load_data(dataset)
    train_df = X_train
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)
    print_graph_statisics(G)
    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir, dataset=dataset)
    save_graph_edgelist(G, dst_dir=dst_dir, dataset=dataset)
    print('Graph saved to disk')

if __name__ == '__main__':
    nyc = "nyc.pkl"
    gowalla = "gowalla.pkl"
    dst_dir = "dataset/tfm/"
    
    build_graph(nyc, dst_dir)
    build_graph(gowalla, dst_dir)
    
