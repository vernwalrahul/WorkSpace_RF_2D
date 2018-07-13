import numpy as np
import networkx as nx
from itertools import chain
import helper
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

no_env = 2000
no_pp = 5
K_nn = 10

def visualize_nodes(curr_occ_grid, curr_node_posns):
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    occ_g = curr_occ_grid.reshape(10,10)
    for i in range(10):
            for j in range(10):
                if(occ_g[i,j]==0):
                    ax1.add_patch(patches.Rectangle(
                    (i/10.0, j/10.0),   # (x,y)
                    0.1,          # width
                    0.1,          # height
                    alpha=0.6
                    ))
    curr_node_posns = np.array(curr_node_posns)
    plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
    plt.title("Visualization")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def get_path_nodes(shallow_G_currP, dense_G, start_n, goal_n, obstacles, curr_occ_grid):
    temp_path_nodes = nx.dijkstra_path(dense_G, start_n, goal_n)
    lmbda_ = [1, 1.5, 2, 5, 10]
    no_nodes = [0, 0, 0, 0, 0]
    THRESHOLD = 0.1

    s_path_nodes = ['o'+node for node in temp_path_nodes]
    for node in temp_path_nodes:
        shallow_G_currP.add_node('o'+node, state = dense_G.node[node]['state'])

    i = 0
    curr_nodes = []
    for lmbda in lmbda_: 
        shallow_G_currP = helper.connect_within_thresh(shallow_G_currP, lmbda, THRESHOLD, s_path_nodes)
        shallow_G_currP = helper.remove_invalid_edges(shallow_G_currP, obstacles)
        curr_nodes = nx.dijkstra_path(shallow_G_currP, 'o'+start_n, 'o'+goal_n)
        curr_node_posns = [helper.state_to_numpy(shallow_G_currP.node[node]['state']) for node in curr_nodes]
        # print("curr_nodes = ", curr_nodes)
        for node in temp_path_nodes:
            if('o'+node in curr_nodes):
                no_nodes[i] += 1
        i += 1

        # visualize_nodes(curr_occ_grid, curr_node_posns)
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    hfont = {'fontname': 'Helvetica'}

    plt.plot(lmbda_, no_nodes, linewidth = 2)
    plt.ylim(0, max(no_nodes)+5)
    plt.xlim(0, max(lmbda_)+5)
    plt.xlabel("Lambda", **hfont)
    plt.ylabel("No of common nodes", **hfont)
    plt.savefig("lmbda_"+`random.randint(0, 100)`+".jpg")
    plt.show()

    
    curr_nodes = [node.strip('o') for node in curr_nodes]

    return curr_nodes

def get_valid_start_goal(dense_G, obstacles, shallow_G_currP):
       
    start_n = random.choice(list(dense_G.nodes()))
    goal_n = random.choice(list(dense_G.nodes()))

    # start_n = '341'
    # goal_n = '665'

    start = helper.state_to_numpy(dense_G.node[start_n]['state'])
    goal = helper.state_to_numpy(dense_G.node[goal_n]['state'])

    while(not helper.valid_start_goal(start, goal, obstacles)):
        start_n = random.choice(list(dense_G.nodes()))
        goal_n = random.choice(list(dense_G.nodes()))

        # start_n = '341'
        # goal_n = '665'

        start = helper.state_to_numpy(dense_G.node[start_n]['state'])
        goal = helper.state_to_numpy(dense_G.node[goal_n]['state'])

    shallow_G_currP.add_node('o'+start_n, state = dense_G.node[start_n]['state'])
    shallow_G_currP.add_node('o'+goal_n, state = dense_G.node[goal_n]['state'])
    # print("added node", 'o'+start_n, 'o'+goal_n)
    shallow_G_currP = helper.connect_knn_for_one_node(shallow_G_currP, K_nn, 'o'+start_n)
    shallow_G_currP = helper.connect_knn_for_one_node(shallow_G_currP, K_nn, 'o'+goal_n)
    
    return start_n, start, goal_n, goal, shallow_G_currP

def append_to_files(start_nodes, goal_nodes, occ_grid, all_path_nodes):
    print("appending to files__")
    assert (len(occ_grid)==len(start_nodes))
    assert (len(all_path_nodes)==len(start_nodes))

    s_file = open("dataset_new/start_nodes.txt",'a')
    g_file = open("dataset_new/goal_nodes.txt",'a')
    occ_file = open("dataset_new/occ_grid.txt", 'a')

    np.savetxt(s_file, np.array(start_nodes), delimiter = " ", fmt = "%s")
    np.savetxt(g_file, np.array(goal_nodes), delimiter = " ", fmt = "%s")
    np.savetxt(occ_file, np.array(occ_grid), delimiter = " ", fmt = "%s")
    helper.write_to_file("dataset_new", all_path_nodes)

def main():
    dense_G = nx.read_graphml("graphs/dense_graph.graphml")
    shallow_G = nx.read_graphml("graphs/shallow_graph.graphml")

    start_nodes = []
    goal_nodes = []
    occ_grid = []
    all_path_nodes = []
    curr_path_nodes = []
    curr_occ_grid = []

    for n in range(no_env):
        print("env_no = ", n)
        obstacles = helper.get_obstacle_posns()
        shallow_G_currE = shallow_G.copy()
        shallow_G_currE = helper.remove_invalid_edges(shallow_G_currE, obstacles)

        dense_G_currE = dense_G.copy()
        dense_G_currE = helper.remove_invalid_edges(dense_G_currE, obstacles)
        curr_occ_grid = helper.get_occ_grid(obstacles)

        for p in range(no_pp):
            print("pp_no = ", p)
            shallow_G_currP = shallow_G_currE.copy()

            start_n, start, goal_n, goal, shallow_G_currP = get_valid_start_goal(dense_G_currE, obstacles, shallow_G_currP)

            flag = False
            while(not flag):
                if(helper.path_exists(dense_G_currE, start_n, goal_n)):
                    if(not helper.path_exists(shallow_G_currP, 'o'+start_n, 'o'+goal_n)):
                        flag = True
                        try:
                            curr_path_nodes = get_path_nodes(shallow_G_currP, dense_G_currE, start_n, goal_n, obstacles, curr_occ_grid)
                            print("curr_path_nodes = ", curr_path_nodes)
                        except Exception as e:
                            print(e)    
                            continue
                    else:
                        print("shallow_G_currP found path")
                        shallow_G_currP.remove_node('o'+start_n)
                        shallow_G_currP.remove_node('o'+goal_n)
                        start_n, start, goal_n, goal, shallow_G_currP = get_valid_start_goal(dense_G_currE, obstacles, shallow_G_currP)
                else:
                    print("no path in dense_G_currE")
                    shallow_G_currP.remove_node('o'+start_n)
                    shallow_G_currP.remove_node('o'+goal_n)
                    start_n, start, goal_n, goal, shallow_G_currP = get_valid_start_goal(dense_G_currE, obstacles, shallow_G_currP)
        
            all_path_nodes.append(curr_path_nodes)
            start_nodes.append(start_n)
            goal_nodes.append(goal_n)
            occ_grid.append(curr_occ_grid)

        if(n%10==0):
            # append_to_files(np.array(start_nodes), np.array(goal_nodes), np.array(occ_grid), all_path_nodes)
            start_nodes = []
            goal_nodes = []
            occ_grid = []
            all_path_nodes = []
if __name__ == '__main__':
    main()