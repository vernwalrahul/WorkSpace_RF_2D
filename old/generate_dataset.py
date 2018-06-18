import numpy as np 
import math
import random
import networkx as nx
import helper
import visualise_training_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rx1 = [2, 12]
rx2 = [None, 16]
rx3 = [2, 16]

ry1 = [2, 16]
ry2 = [8, 18]
ry3 = [2, None]

EDGE_DISCRETIZATION = 11

def calc_weight(config1, config2):
    return math.sqrt(float(np.sum((config2-config1)**2)))

def path_exists(src, goal, G):
    try:
        paths = nx.dijkstra_path(G, src, goal, weight='weight')
        return 1
    except Exception as e:
        # print("Invalid start-goal-------------------------------------------------------------------")
        return 0    

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

def is_trivial(start, goal, obstacles):
    start = np.array(start)
    goal = np.array(goal)

    diff = goal - start
    step = diff/EDGE_DISCRETIZATION

    for i in range(EDGE_DISCRETIZATION+1):
        nodepos = start + step*i
        if(not (helper.is_valid(nodepos, obstacles))):
            return 0

    return 1        

def get_obstacles_posns():
    x1 = random.randint(rx1[0], rx1[1])
    rx2[0] = x1+4

    x2 = random.randint(rx2[0], rx2[1])
    x3 = random.randint(rx3[0], rx3[1])

    y1 = random.randint(ry1[0], ry1[1])
    y2 = random.randint(ry2[0], ry2[1])
    ry3[1] = y2-4

    y3 = random.randint(ry3[0], ry3[1])

    x1/=20.0
    x2/=20.0
    x3/=20.0
    y1/=20.0
    y2/=20.0
    y3/=20.0
    obs1 = (0, y1, x1, y1+0.1)
    obs2 = (x1+0.05, y1, x2, y1+0.1)
    obs3 = (x2+0.05, y1, 1, y1+0.1)

    obs4 = (x3, y2, x3+0.1, 1)
    obs5 = (x3, y3, x3+0.1, y2-0.05)
    obs6 = (x3, 0, x3+0.1, y3-0.05)

    return [obs1, obs2, obs3, obs4, obs5, obs6]

def get_occ_grid(obstacles):
    occ_grid1 = np.ones((20,20), dtype=int)
    eps = 0.025
    for i in range(0,20):
        for j in range(0, 20):
            if(not (helper.is_valid((i/20.0+eps,j/20.0+eps), obstacles))):
                occ_grid1[i,j] = 0
            else:
                occ_grid1[i,j] = 1
    return occ_grid1.ravel()

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

        # if(check_for_collision(node, sn)==1):
        G.add_edge(node, sn)
        # print("connected edge from ",node, " to ",sn)
        G[node][sn]['weight'] = w
        G1.remove_node(sn)
    return G

def connect_knn(shallow_G1, dense_G1, start_n, goal_n, k):
    shallow_G1.add_node('o'+start_n, state = dense_G1.node[start_n]['state'])
    shallow_G1.add_node('o'+goal_n, state = dense_G1.node[goal_n]['state'])

    shallow_G1 = connect_knn_for_one_node(shallow_G1, k, 'o'+start_n)
    shallow_G1 = connect_knn_for_one_node(shallow_G1, k, 'o'+goal_n)

    return shallow_G1

def find_closest_node(shallow_G1, node_posn):
    dist = 10000
    c_node = None

    for node in list(shallow_G1.nodes()):
        pos = state_to_numpy(shallow_G1.node[node]['state'])
        if(calc_weight(pos, node_posn)<dist):
            dist = calc_weight(pos, node_posn)
            c_node = node
    return c_node, dist

def is_in_collision(prev_posn, curr_posn, obstacles):
    start = np.array(prev_posn)
    goal = np.array(curr_posn)

    diff = goal - start
    step = diff/EDGE_DISCRETIZATION

    for i in range(EDGE_DISCRETIZATION+1):
        nodepos = start + step*i
        if(not (helper.is_valid(nodepos, obstacles))):
            return 1

    return 0


def get_path_nodes(shallow_G1, dense_G1, start_n, goal_n, obstacles):
    c_path_nodes = []

    threshold = 0.05

    prev = 'o'+start_n
    prev_posn = state_to_numpy(shallow_G1.node[prev]['state'])
    
    temp_prev = start_n
    temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])

    temp_curr = start_n
    temp_curr_posn = state_to_numpy(dense_G1.node[temp_curr]['state'])

    curr = 'o'+start_n
    curr_posn = state_to_numpy(shallow_G1.node[curr]['state'])

    temp_path_nodes = nx.dijkstra_path(dense_G1, start_n, goal_n)
    curr_dist = 0
    i = 0

    while(i<len(temp_path_nodes)-1):
        i += 1
        prev = curr
        prev_posn = curr_posn

        temp_curr = temp_path_nodes[i]
        temp_curr_posn = state_to_numpy(dense_G1.node[temp_curr]['state'])
        curr, dist = find_closest_node(shallow_G1, state_to_numpy(dense_G1.node[temp_curr]['state']))
        curr_posn = state_to_numpy(shallow_G1.node[curr]['state'])

        if(dist>threshold):
            d = calc_weight(temp_curr_posn, prev_posn)
            if(d>threshold):
                c_path_nodes.append(temp_prev)
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                curr = temp_curr
                curr_posn = temp_curr_posn
                continue
            else:
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                curr = temp_curr
                curr_posn = temp_curr_posn
                continue    

        if(is_in_collision(temp_curr_posn, curr_posn, obstacles)):
            if(is_in_collision(temp_curr_posn, prev_posn, obstacles)):
                c_path_nodes.append(temp_prev)
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                curr = temp_curr
                curr_posn = temp_curr_posn
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                continue
            else:
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                curr = temp_curr
                curr_posn = temp_curr_posn
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                continue

        if(is_in_collision(prev_posn, curr_posn, obstacles)):
            if(is_in_collision(temp_curr_posn, prev_posn, obstacles)):
                c_path_nodes.append(temp_prev)
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                curr = temp_curr
                curr_posn = temp_curr_posn
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                continue
            else:
                c_path_nodes.append(temp_curr)
                temp_prev = temp_curr
                curr = temp_curr
                curr_posn = temp_curr_posn
                temp_prev_posn = state_to_numpy(dense_G1.node[temp_prev]['state'])
                continue
    if(len(c_path_nodes)==0):
        c_path_nodes = ['-1']            
    return c_path_nodes

def plot_occ_grid(occ_grid):
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i in range(20):
        for j in range(20):
            if(occ_grid[i,j]==0):
                ax1.add_patch(patches.Rectangle(
                (j/20.0, i/20.0),   # (x,y)
                0.05,          # width
                0.05,          # height
                alpha=0.6
                ))
    plt.show()            

def main():
    random.seed(1000)
    dense_G = nx.read_graphml("graphs_2d/dense_graph.graphml")
    shallow_G = nx.read_graphml("graphs_2d/shallow_graph.graphml")
    start_nodes = []
    goal_nodes = []

    occ_grid = []
    all_path_nodes = []
    no_env = 1
    no_pp = 5
    no_paths = 5
    knn = 10

    for n in range(no_env):
        print("----------------------------------------------env_no = ",n)
        dense_G1 = dense_G.copy()
        shallow_G1 = shallow_G.copy()
        obstacles = get_obstacles_posns()
        print("obstacles = ", obstacles)
        occ_grid1 = get_occ_grid(obstacles)
        dense_G1 = helper.remove_invalid_edges(dense_G1, obstacles)

        for p in range(no_pp):
            print("pp_no = ", p)
            flag = False
            while(flag==False):
                start_n = random.choice(list(dense_G1.nodes()))
                goal_n = random.choice(list(dense_G1.nodes()))

                start = state_to_numpy(dense_G1.node[start_n]['state'])
                goal = state_to_numpy(dense_G1.node[goal_n]['state'])
                if(not is_trivial(start, goal, obstacles)):
                    if(path_exists(start_n, goal_n, dense_G1)):
                        shallow_G1.add_node(start_n, state = dense_G1.node[start_n]['state'])
                        shallow_G1.add_node(goal_n, state = dense_G1.node[goal_n]['state'])

                        shallow_G1 = connect_knn(shallow_G1, dense_G1, start_n, goal_n, knn)
                        if(not path_exists(shallow_G1, 'o'+start_n, 'o'+goal_n)):
                            start_nodes.append(start_n)
                            goal_nodes.append(goal_n)
                            occ_grid.append(occ_grid1)
                            # print("occ_grid = ", occ_grid1.reshape(20,20))
                            # plot_occ_grid(occ_grid1.reshape(20,20))
                            # return
                            flag = True
                            path_nodes = get_path_nodes(shallow_G1, dense_G1, start_n, goal_n, obstacles)
                            all_path_nodes.append(path_nodes)
                            print("path_nodes = ",path_nodes)
                            continue


    occ_grid = np.array(occ_grid)
    print("occ_grid.shape = ",occ_grid.shape)
    np.savetxt("dataset/start_nodes.txt", np.array(start_nodes), delimiter = " ", fmt = "%s")
    np.savetxt("dataset/goal_nodes.txt", np.array(goal_nodes), delimiter = " ", fmt = "%s")
    np.savetxt("dataset/occ_grid.txt", np.array(occ_grid), delimiter = " ", fmt = "%s")
    helper.write_to_file("dataset", all_path_nodes)


if __name__ == '__main__':
    main()
