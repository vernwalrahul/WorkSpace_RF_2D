import numpy as np
import networkx as nx
from itertools import chain
import helper
import random

no_env = 20
no_pp = 5

def append_to_files(start_posns, goal_posns, occ_grid, pp_type):
    assert (len(occ_grid)==len(start_posns))
    assert (len(goal_posns)==len(start_posns))

    s_file = open("test_dataset_23June/"+pp_type+"/start_posns.txt",'w')
    g_file = open("test_dataset_23June/"+pp_type+"/goal_posns.txt",'w')
    occ_file = open("test_dataset_23June/"+pp_type+"/occ_grid.txt", 'w')

    np.savetxt(s_file, np.array(start_posns), delimiter = " ", fmt = "%s")
    np.savetxt(g_file, np.array(goal_posns), delimiter = " ", fmt = "%s")
    np.savetxt(occ_file, np.array(occ_grid), delimiter = " ", fmt = "%s")

def get_easy_pp(obstacles, start_posns, goal_posns, occ_grid, x1, x2, x3, y1, y2):

	while(True):
		sx = random.random()*x3
		sy = y1+0.1 + random.random()*(0.9-y1)

		if(random.randint(0,100)%2==0):
			gx = x3 + 0.1 + random.random()*(0.9-x3)
			gy = y1 + 0.1 + random.random()*(0.9-y1)
		else:
			gx = random.random()*x3
			gy = random.random()*y1

		if(not (helper.is_free((sx, sy), obstacles) and helper.is_free((gx, gy), obstacles))):
			continue

		x = random.randint(0,100)

		if(x%2==0):
			start_posns.append([sx, sy])
			goal_posns.append([gx, gy])
		else:
			start_posns.append([gx, gy])
			goal_posns.append([sx, sy])

		return start_posns, goal_posns, occ_grid

def get_medium_pp(obstacles, start_posns, goal_posns, occ_grid, x1, x2, x3, y1, y2):

	while(True):
		if(random.randint(0,100)%2==0):
			sx = random.random()*x3
			sy = random.random()*y1
			gx = x3 + 0.1 + random.random()*(0.9-x3)
			gy = y1 + 0.1 + random.random()*(0.9-y1)
		else:
			sx = random.random()*x3
			sy = y1+0.1 + random.random()*(0.9-y1)
			gx = x3 + 0.1 + random.random()*(0.9-x3)
			gy = random.random()*y1

		if(not (helper.is_free((sx, sy), obstacles) and helper.is_free((gx, gy), obstacles))):
				continue

		x = random.randint(0,100)

		if(x%2==0):
			start_posns.append([sx, sy])
			goal_posns.append([gx, gy])
		else:
			start_posns.append([gx, gy])
			goal_posns.append([sx, sy])

		return start_posns, goal_posns, occ_grid

def get_hard_pp(obstacles, start_posns, goal_posns, occ_grid, x1, x2, x3, y1, y2):

	while(True):
		sx = random.random()*x3
		sy = random.random()*y1
		gx = x3 + 0.1 + random.random()*(0.9-x3)
		gy = random.random()*y1

		if(not (helper.is_free((sx, sy), obstacles) and helper.is_free((gx, gy), obstacles))):
				continue

		x = random.randint(0,100)

		if(x%2==0):
			start_posns.append([sx, sy])
			goal_posns.append([gx, gy])
		else:
			start_posns.append([gx, gy])
			goal_posns.append([sx, sy])

		return start_posns, goal_posns, occ_grid

def main():

	start_posns = []
	goal_posns = []
	occ_grid = []
	sx, sy, gx, gy = 0, 0, 0, 0

	pp_type = "easy"

	for n in range(no_env):
		obstacles = helper.get_obstacle_posns()

		x1, x2, x3, y1, y2 = obstacles[0][2], obstacles[1][2], obstacles[3][0], obstacles[0][1], obstacles[4][3] 
		curr_occ_grid = helper.get_occ_grid(obstacles)

		for p in range(no_pp):

			start_posns, goal_posns, occ_grid = get_easy_pp(obstacles, start_posns, goal_posns, occ_grid, x1, x2, x3, y1, y2)
			occ_grid.append(curr_occ_grid)

	start_posns = np.array(start_posns)
	goal_posns = np.array(goal_posns)
	occ_grid = np.array(occ_grid)

	append_to_files(np.array(start_posns), np.array(goal_posns), np.array(occ_grid), pp_type)



if __name__ == '__main__':
	main()