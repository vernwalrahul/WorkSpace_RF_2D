import numpy as np
import networkx as nx
import random
import math

rx1 = [1, 6]
rx2 = [None, 8]
rx3 = [1, 8]

ry1 = [1, 8]
ry2 = [4, 9]
ry3 = [1, None]
EDGE_DISCRETIZATION_T = 40
EDGE_DISCRETIZATION = 11 
THRESHOLD = 0.2

def write_to_file(directory, all_paths):
    with open(directory + "/path_nodes.txt", 'w') as file:
        file.writelines(','.join(str(j) for j in i) + '\n' for i in all_paths)
        
def calc_weight(config1, config2):
    return math.sqrt(float(np.sum((config2-config1)**2)))

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

#check for existense of a feasible path
def path_exists(G, src, goal):
    try:
        paths = nx.dijkstra_path(G, src, goal, weight='weight')
        return 1
    except Exception as e:
        return 0

#to return random obstacle posns
def get_obstacle_posns():
    # x = x+2
    # obs1 = [0.0, 0.1, 0.6, 0.1+0.1]
    # obs2 = [0.6+0.1, 0.1, 0.8, 0.1+0.1]
    # obs3 = [0.9, 0.1, 1.0, 0.1+0.1]

    # obs4 = [0.7, x/10.0, 0.7+0.1, 1.0]
    # obs5 = [0.7, 0.1, 0.7+0.1, x/10.0-0.1]
    # obs6 = [0.7, 0.0, 0.7+0.1, 0.1]
    # return [obs1, obs2, obs3, obs4, obs5, obs6]


    x1 = random.randint(rx1[0], rx1[1])
    rx2[0] = x1+2

    x2 = random.randint(rx2[0], rx2[1])
    x3 = random.randint(rx3[0], rx3[1])

    y1 = random.randint(ry1[0], ry1[1])
    y2 = random.randint(ry2[0], ry2[1])
    ry3[1] = y2-2

    y3 = random.randint(ry3[0], ry3[1])

    x1/=10.0
    x2/=10.0
    x3/=10.0
    y1/=10.0
    y2/=10.0
    y3/=10.0
    obs1 = (0, y1, x1, y1+0.1)
    obs2 = (x1+0.1, y1, x2, y1+0.1)
    obs3 = (x2+0.1, y1, 1, y1+0.1)

    obs4 = (x3, y2, x3+0.1, 1)
    obs5 = (x3, y3, x3+0.1, y2-0.1)
    obs6 = (x3, 0, x3+0.1, y3-0.1)

    return [obs1, obs2, obs3, obs4, obs5, obs6]

#to check if a node is free
def is_free(node_pos, obstacles):
    flag = 1
    eps = 0.04
    for obs in obstacles:
        x1, y1, x2, y2 = obs
        if(node_pos[0] < x2 + eps and node_pos[0] > x1 - eps):
            if(node_pos[1] < y2 + eps and node_pos[1] > y1 - eps):
                flag = 0
                return flag
    return flag

#check for the planning  problem being trivial
def valid_start_goal(start, goal, obstacles):
    start = np.array(start)
    goal = np.array(goal)

    if(not (is_free(start, obstacles) and is_free(goal, obstacles))):
        # print("start_free = ",is_free(start, obstacles))
        # print("goal_free = ",is_free(goal, obstacles))
        return 0

    diff = goal - start
    step = diff/EDGE_DISCRETIZATION_T

    for i in range(EDGE_DISCRETIZATION_T+1):
        nodepos = start + step*i
        if(not (is_free(nodepos, obstacles))):
            return 1
    # print("trivial")        
    return 0

#to check if two nodes are within threshold and can be connected
def satisfy_condition(node1_pos, node2_pos, obstacles):
    node1_pos, node2_pos = np.array(node1_pos), np.array(node2_pos)
    if(calc_weight(node1_pos, node2_pos)>THRESHOLD):
        return 0
    
    diff = node2_pos - node1_pos
    step = diff/EDGE_DISCRETIZATION

    for i in range(EDGE_DISCRETIZATION+1):
        nodepos = node1_pos + step*i
        if(not (is_free(nodepos, obstacles))):
            return 0

    return 1

#return occupancy grid
def get_occ_grid(obstacles):
    occ_grid = np.ones((10,10), dtype=int)
    eps = 0.05
    for i in range(0,10):
        for j in range(0, 10):
            if(not (is_free((i/10.0+eps,j/10.0+eps), obstacles))):
                occ_grid[i,j] = 0
            else:
                occ_grid[i,j] = 1
    return occ_grid.ravel()

#connect knn 
def connect_knn_for_one_node(G, K, node):
    state = G.node[node]['state']
    conf = state_to_numpy(state)
    G1 = G.copy()

    for k in range(K):
        w = 1000000
        sn = None
        for node1 in G1.nodes():
            if(node == node1):
                continue
            state1 = G1.node[node1]['state']
            conf1  = state_to_numpy(state1)
            if(calc_weight(conf, conf1) < w):
                w = calc_weight(conf, conf1)
                sn = node1
        if(w<THRESHOLD):
            G.add_edge(node, sn)
            G[node][sn]['weight'] = w
            G1.remove_node(sn)
        else:
            break    
    return G

#closest node to a point
def find_closest_node(shallow_G1, node_posn):
    dist = 10000
    c_node = None

    for node in list(shallow_G1.nodes()):
        pos = state_to_numpy(shallow_G1.node[node]['state'])
        if(calc_weight(pos, node_posn)<dist):
            dist = calc_weight(pos, node_posn)
            c_node = node
    
    return c_node

def remove_invalid_edges(G, obstacles):
    to_remove = []
    for i,edge in enumerate(G.edges()):
        u,v = edge
        node1_pos = state_to_numpy(G.node[u]['state'])
        node2_pos = state_to_numpy(G.node[v]['state'])

        diff = node2_pos - node1_pos
        step = diff/EDGE_DISCRETIZATION

        for i in range(EDGE_DISCRETIZATION+1):
            nodepos = node1_pos + step*i
            if(not (is_free(nodepos, obstacles))):
                to_remove.append((u,v))
                break
    
    for edge in to_remove:
        u, v = edge
        G.remove_edge(u, v) 
    return G