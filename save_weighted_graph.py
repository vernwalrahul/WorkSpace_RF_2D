import os
import numpy as np
import math
import openravepy
import argparse
import sys
import json
import networkx as nx

def calc_weight(config1, config2):
    return math.sqrt(float(np.sum((config2-config1)**2)))

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list) 

def save_modified_graph(G):
    file_addr = "graphs_2d/dense_graph.graphml"
    to_remove = []
    for i, edge in enumerate(G.edges()):
        u, v = edge
        G[u][v]['weight'] = calc_weight(state_to_numpy(G.node[u]['state']), state_to_numpy(G.node[v]['state']))
        if(G[u][v]['weight']>0.1):
            to_remove.append([u, v])
    for edge in to_remove:
        u, v = edge
        G.remove_edge(u, v)        
    nx.write_graphml(G, file_addr)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate environments')
    parser.add_argument('--graphfile',type=str,required=True)
    args = parser.parse_args()
    
    G = nx.read_graphml(args.graphfile)

    save_modified_graph(G)
