import pickle
import pandas as pd


# Check the contents of gowalla_graph.pkl
print("\nContents of gowalla_graph.pkl:")
with open("dataset/tfm/gowalla_graph.pkl", 'rb') as f:
    graph_data = pickle.load(f)

    # Print the type of the loaded object
    print("Type of graph_data:", type(graph_data))

    # Print the attributes of the graph_data object
    print("\nAttributes of graph_data:")
    for attr in dir(graph_data):
        if not attr.startswith('_'):
            print(attr)

    # Print the number of nodes and edges in the graph
    print("\nNumber of nodes:", graph_data.number_of_nodes())
    print("Number of edges:", graph_data.number_of_edges())
    #print("nodes:", graph_data.nodes())
    #print("adj:", graph_data.adj)   