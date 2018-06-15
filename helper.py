import networkx as nx 
import numpy as np
import argparse
from random import choice
from itertools import islice, chain
import math

EDGE_DISCRETIZATION = 11

def calc_weight(config1, config2):
    return math.sqrt(float(np.sum((config2-config1)**2)))

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

def edge_to_configs(state1, state2):
    config1 = state_to_numpy(state1)
    config2 = state_to_numpy(state2)

    diff = config2 - config1
    step = diff/EDGE_DISCRETIZATION

    to_check = list()
    to_check.append(config1)

    for i in xrange(EDGE_DISCRETIZATION - 1):
        conf = config1 + step*(i+1)
        to_check.append(conf)

    return to_check

def is_valid(node_pos, obstacles):
    flag = 1
    for obs in obstacles:
        x1, y1, x2, y2 = obs
        if(node_pos[0] < x2 and node_pos[0] > x1):
            if(node_pos[1] < y2 and node_pos[1] > y1):
                flag = 0
                return flag
    return flag

def remove_invalid_edges(G, obstacles):
    to_remove = []
    for i,edge in enumerate(G.edges()):
        u,v = edge
        state1 = G.node[u]['state']
        state2 = G.node[v]['state']
        configs_to_check = edge_to_configs(state1,state2) 

        edge_free = 1
        for cc in configs_to_check:
            if(not is_valid( cc, obstacles )):
                edge_free = 0
                break
        if(not edge_free):
            to_remove.append([u, v])
    
    for edge in to_remove:
        u, v = edge
        G.remove_edge(u, v) 
    return G                             

def write_to_file(directory, all_paths):
    with open(directory + "/path_nodes.txt", 'w') as file:
        file.writelines(','.join(str(j) for j in i) + '\n' for i in all_paths)

def remove_one_edge(G, src, goal):
    threshold = 10
    n = 15
    Edge_Count = {}
    all_path_nodes = list(islice(nx.shortest_simple_paths(G, src, goal, weight='weight'), n))

    max_v, m_key = 0, None
    for path_nodes in all_path_nodes:
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if(v<u):
                temp = u
                u = v
                v = temp
            key = str(u)+","+str(v) 
            if(key in Edge_Count):
                Edge_Count[key] += 1
            else:
                Edge_Count[key] = 1
            for key in Edge_Count:
                if(Edge_Count[key]>max_v):
                    max_v = Edge_Count[key]
                    m_key = key

    u, v = m_key.split(",")
    # print("removing edge = ", m_key, " count = ",max_v)     
    G.remove_edge(u, v)
    return G

def get_robust_shortest_paths(G, src, goal, K):
    path_nodes_all = []
    for k in range(K):
        try:
            path_nodes_all.append((nx.dijkstra_path(G, src, goal, weight='weight')))
            # print("k =", k, "dijkstra_path = ", path_nodes_all[-1])
            path_length = 0
            for i in range(len(path_nodes_all[-1])-1):
                config1 = state_to_numpy(G.node[path_nodes_all[-1][i]]['state'])
                config2 = state_to_numpy(G.node[path_nodes_all[-1][i+1]]['state'])
                path_length += calc_weight(config1, config2)
            # print("path_length = ", path_length)    

            G = remove_one_edge(G, src, goal)
        except Exception as e:
            # print(e)    
            path_nodes_all.append([-1])
    return list(chain.from_iterable(path_nodes_all))