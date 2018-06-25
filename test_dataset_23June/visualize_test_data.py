import numpy as np
import networkx as nx
from itertools import chain
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list) 

def main():
    parser = argparse.ArgumentParser(description='Generate environments')
    parser.add_argument('--envdir',type=str,required=True)
    args = parser.parse_args()

    start_posns = np.loadtxt(args.envdir + "/start_posns.txt", delimiter = " ")
    goal_posns = np.loadtxt(args.envdir + "/goal_posns.txt", delimiter = " ")
    occ_grid = np.loadtxt(args.envdir + "/occ_grid.txt", delimiter = " ")

    n = len(start_posns)

    i = random.randint(0, n-1)

    src, goal, og = start_posns[i], goal_posns[i], occ_grid[i].reshape(10,10)

    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i in range(10):
        for j in range(10):
            if(og[i,j]==0):
                ax1.add_patch(patches.Rectangle(
                (i/10.0,j/10.0),   # (x,y)
                0.1,          # width
                0.1,          # height
                alpha=0.6
                ))

    plt.scatter(src[0], src[1], color = "red",  s = 100, edgecolors="black")
    plt.scatter(goal[0], goal[1], color = "blue",  s = 100, edgecolors="black")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

if __name__ == '__main__':
    main()