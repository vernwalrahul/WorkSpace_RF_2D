import numpy as np
import networkx as nx
from itertools import chain
import helper
import random

no_env = 2000
no_pp = 5
K_nn = 10

def get_path_nodes(shallow_G_currP, dense_G, start_n, goal_n, obstacles):
    temp_path_nodes = nx.dijkstra_path(dense_G, start_n, goal_n)
    path_nodes = []

    prev = curr = 'o'+start_n
    prev_posn = curr_posn = helper.state_to_numpy(shallow_G_currP.node[curr]['state'])
    
    prev_temp = curr_temp = start_n
    prev_temp_posn = curr_temp_posn = helper.state_to_numpy(dense_G.node[curr_temp]['state'])

    i = 0
    while(i<len(temp_path_nodes)-1):
        i += 1

        curr_temp = temp_path_nodes[i]
        curr_temp_posn = helper.state_to_numpy(dense_G.node[curr_temp]['state'])

        curr = helper.find_closest_node(shallow_G_currP, curr_temp_posn)
        curr_posn = helper.state_to_numpy(shallow_G_currP.node[curr]['state'])

        if(curr==prev):
            continue

        if(helper.satisfy_condition(curr_posn, curr_temp_posn, obstacles)):
            if(helper.satisfy_condition(curr_posn, prev_posn, obstacles)):
                prev_temp = curr_temp
                prev_temp_posn = curr_temp_posn
                prev = curr
                prev_posn = curr_posn
                continue
            elif(helper.satisfy_condition(prev_posn, curr_temp_posn, obstacles)):
                path_nodes.append(curr_temp)
                prev_temp = curr_temp
                prev = 'o'+curr_temp
                shallow_G_currP.add_node(prev, state = dense_G.node[curr_temp]['state'])
                prev_temp_posn = prev_posn = curr_temp_posn
                continue
            else:
                path_nodes.append(prev_temp)
                path_nodes.append(curr_temp)
                shallow_G_currP.add_node('o'+prev_temp, state = dense_G.node[prev_temp]['state'])
                shallow_G_currP.add_node('o'+curr_temp, state = dense_G.node[curr_temp]['state'])
                prev_temp = curr_temp
                prev = 'o'+curr_temp

                prev_temp_posn = prev_posn = curr_temp_posn
                continue
        else:
            if(helper.satisfy_condition(prev_posn, curr_temp_posn, obstacles)):
                path_nodes.append(curr_temp)
                prev_temp = curr_temp
                prev = 'o'+curr_temp
                shallow_G_currP.add_node(prev, state = dense_G.node[curr_temp]['state'])
                prev_temp_posn = prev_posn = curr_temp_posn
                continue
            else:
                path_nodes.append(prev_temp)
                path_nodes.append(curr_temp)
                shallow_G_currP.add_node('o'+prev_temp, state = dense_G.node[prev_temp]['state'])
                shallow_G_currP.add_node('o'+curr_temp, state = dense_G.node[curr_temp]['state'])
                prev_temp = curr_temp
                prev = 'o'+curr_temp

                prev_temp_posn = prev_posn = curr_temp_posn
                continue

    assert (len(path_nodes)>0)          
    return path_nodes

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
                            curr_path_nodes = get_path_nodes(shallow_G_currP, dense_G_currE, start_n, goal_n, obstacles)
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

    assert (len(occ_grid)==len(start_nodes))
    assert (len(all_path_nodes)==len(start_nodes))

    np.savetxt("dataset/start_nodes.txt", np.array(start_nodes), delimiter = " ", fmt = "%s")
    np.savetxt("dataset/goal_nodes.txt", np.array(goal_nodes), delimiter = " ", fmt = "%s")
    np.savetxt("dataset/occ_grid.txt", np.array(occ_grid), delimiter = " ", fmt = "%s")
    helper.write_to_file("dataset", all_path_nodes)

if __name__ == '__main__':
    main()