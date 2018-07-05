import matplotlib.pyplot as plt
import numpy as np

dense_path_length = np.loadtxt("dense_path_lengths.txt")

p5_rf1_path_length_matrix = np.loadtxt("p5/RF1_pl_matrix_lmbda1.txt")
p5_rf2_path_length_matrix = np.loadtxt("p5/RF2_pl_matrix_lmbda1.txt")
p5_rf3_path_length_matrix = np.loadtxt("p5/RF3_pl_matrix_lmbda1.txt")

p5_rf_path_length_matrix = np.zeros(p5_rf1_path_length_matrix.shape)

p5_sd_rf = np.zeros(p5_rf1_path_length_matrix.shape[0])
p7_sd_rf = np.zeros(p5_rf1_path_length_matrix.shape[0])
sd_sp =  np.zeros(p5_rf1_path_length_matrix.shape[0])

p5_pl_rf = np.zeros(p5_rf1_path_length_matrix.shape[0])
p7_pl_rf = np.zeros(p5_rf1_path_length_matrix.shape[0])
pl_sp =  np.zeros(p5_rf1_path_length_matrix.shape[0])
pl_halton = np.zeros(p5_rf1_path_length_matrix.shape[0])

count = 0

for i in range(p5_rf_path_length_matrix.shape[0]):
	for j in range(p5_rf_path_length_matrix.shape[1]):
		flag = 1
		if(p5_rf1_path_length_matrix[i,j]==-1):
			p5_rf1_path_length_matrix[i,j] = 0
			flag = 0
		if(p5_rf2_path_length_matrix[i,j]==-1):
			p5_rf2_path_length_matrix[i,j] = 0
			flag = 0
		if(p5_rf3_path_length_matrix[i,j]==-1):
			p5_rf3_path_length_matrix[i,j] = 0
			flag = 0
		p5_rf_path_length_matrix[i,j] = flag

for i in range(p5_rf_path_length_matrix.shape[0]):
	for j in range(p5_rf_path_length_matrix.shape[1]):
		if(p5_rf_path_length_matrix[i,j]==0):
			p5_rf_path_length_matrix[i,j] = -1
			continue
		p5_rf_path_length_matrix[i,j] = (p5_rf1_path_length_matrix[i,j] + p5_rf2_path_length_matrix[i,j] + p5_rf3_path_length_matrix[i,j])/3.0



p7_rf1_path_length_matrix = np.loadtxt("p7/RF1_pl_matrix_lmbda1.txt")
p7_rf2_path_length_matrix = np.loadtxt("p7/RF2_pl_matrix_lmbda1.txt")
p7_rf3_path_length_matrix = np.loadtxt("p7/RF3_pl_matrix_lmbda1.txt")

p7_rf_path_length_matrix = np.zeros(p7_rf1_path_length_matrix.shape)

for i in range(p7_rf_path_length_matrix.shape[0]):
	for j in range(p7_rf_path_length_matrix.shape[1]):
		flag = 1
		if(p7_rf1_path_length_matrix[i,j]==-1):
			p7_rf1_path_length_matrix[i,j] = 0
			flag = 0
		if(p7_rf2_path_length_matrix[i,j]==-1):
			p7_rf2_path_length_matrix[i,j] = 0
			flag = 0
		if(p7_rf3_path_length_matrix[i,j]==-1):
			p7_rf3_path_length_matrix[i,j] = 0
			flag = 0
		p7_rf_path_length_matrix[i,j] = flag

for i in range(p7_rf_path_length_matrix.shape[0]):
	for j in range(p7_rf_path_length_matrix.shape[1]):
		if(p7_rf_path_length_matrix[i,j]==0):
			p7_rf_path_length_matrix[i,j] = -1
			continue
		p7_rf_path_length_matrix[i,j] = (p7_rf1_path_length_matrix[i,j] + p7_rf2_path_length_matrix[i,j] + p7_rf3_path_length_matrix[i,j])/3.0


halton_path_length_matrix = np.loadtxt("Halton_pl_matrix_lmbda1.txt")

sp1_path_length_matrix = np.loadtxt("SP1_pl_matrix_lmbda1.txt")
sp2_path_length_matrix = np.loadtxt("SP2_pl_matrix_lmbda1.txt")
sp3_path_length_matrix = np.loadtxt("SP3_pl_matrix_lmbda1.txt")

sp_path_length_matrix = np.zeros(sp1_path_length_matrix.shape)
for i in range(sp_path_length_matrix.shape[0]):
	for j in range(sp_path_length_matrix.shape[1]):
		flag = 1
		if(sp1_path_length_matrix[i,j]==-1):
			sp1_path_length_matrix[i,j] = 0
			flag = 0
		if(sp2_path_length_matrix[i,j]==-1):
			sp2_path_length_matrix[i,j] = 0
			flag = 0
		if(sp3_path_length_matrix[i,j]==-1):
			sp3_path_length_matrix[i,j] = 0
			flag = 0
		sp_path_length_matrix[i,j] = flag

for i in range(sp_path_length_matrix.shape[0]):
	for j in range(sp_path_length_matrix.shape[1]):
		if(sp_path_length_matrix[i,j]==0):
			sp_path_length_matrix[i,j] = -1
			continue
		sp_path_length_matrix[i,j] = (sp1_path_length_matrix[i,j] + sp2_path_length_matrix[i,j] + sp3_path_length_matrix[i,j])/3.0


n = [200, 400, 600, 800, 1000, 1200]

sr_p5_rf = [0, 0, 0, 0, 0, 0]
sr_p7_rf = [0, 0, 0, 0, 0, 0]
sr_halton = [0, 0, 0, 0, 0, 0]
sr_sp = [0, 0, 0, 0, 0, 0,]

hfont = {'fontname': 'Helvetica'}

from math import sqrt

# cases = [2, 5, 6, 7, 8, 9, 11, 12, 14, 26, 28, 35, 36, 37, 38, 39, 70, 72, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
cases = [85, 86, 87, 88]
def calc_sd(a, b, c):
	mean = (a+b+c)/3.0
	sd = sqrt((a-mean)**2+(b-mean)**2+(c-mean)**2)
	return sd

count = np.zeros(len(n))
consider = {}

for i in range(len(n)):
	consider[i] = []
	for j in range(len(p5_rf_path_length_matrix[i])):
		# if(p5_rf_path_length_matrix[i,j]==-1 or p7_rf_path_length_matrix[i,j]==-1 or sp_path_length_matrix[i,j]==-1 or halton_path_length_matrix[i,j]==-1):
		# 	continue
		if (not j in cases):
			continue
		consider[i].append(j)
		p5_sd_rf[i] += calc_sd(p5_rf1_path_length_matrix[i,j], p5_rf2_path_length_matrix[i,j], p5_rf3_path_length_matrix[i,j])
		p7_sd_rf[i] += calc_sd(p7_rf1_path_length_matrix[i,j], p7_rf2_path_length_matrix[i,j], p7_rf3_path_length_matrix[i,j])
		sd_sp[i] += calc_sd(sp1_path_length_matrix[i,j], sp2_path_length_matrix[i,j], sp3_path_length_matrix[i,j])

		p5_pl_rf[i] += p5_rf_path_length_matrix[i,j]/dense_path_length[j]
		p7_pl_rf[i] += p7_rf_path_length_matrix[i,j]/dense_path_length[j]
		pl_sp[i] += sp_path_length_matrix[i,j]/dense_path_length[j]
		pl_halton[i] = halton_path_length_matrix[i,j]/dense_path_length[j]
		# print("i = ",i," adding p7_pl_rf = ", p7_pl_rf[i])

		count[i] += 1

for i in range(len(n)):
	p5_sd_rf[i]/=count[i]
	p7_sd_rf[i]/=count[i]
	sd_sp[i]/=count[i]

	p5_pl_rf[i]/=count[i]
	p7_pl_rf[i]/=count[i]
	pl_sp[i]/=count[i]

print("pl_sp = ", pl_sp)
print("p5_pl_rf = ", p5_pl_rf)
print("p7_pl_rf = ", p7_pl_rf)

# plt.plot(n[1:], p5_pl_rf[1:], color = "green", linewidth = 2, label = "RF+Halton(50:50)")
plt.plot(n[1:], p7_pl_rf[1:], color = "orange", linewidth = 2, label = "RF+Halton(30:70)")
# plt.fill_between(n[1:], p7_pl_rf[1:]-p7_sd_rf[1:], p7_pl_rf[1:]+p7_sd_rf[1:], color = 'orange', alpha = 0.5)
plt.plot(n[1:], pl_sp[1:], color = "blue", linewidth = 2, label = "SP+Halton")
# plt.fill_between(n[1:], pl_sp[1:]-sd_sp[1:], pl_sp[1:]+sd_sp[1:], color = 'blue', alpha = 0.5)
plt.plot(n[1:], pl_halton[1:], color = "red", linewidth = 2, label = "Halton")

plt.xlabel("No of Samples ", **hfont)
plt.ylabel("Cost (Normalized) ", **hfont)

leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize='medium')

plt.xlim(0, 2000)
plt.ylim(0.8, 1.2)
plt.title("Path Length", **hfont)
plt.grid(True)
plt.savefig("Path_Length.jpg", bbox_inches='tight')
plt.show()

def intersection(lst1, lst2, lst3, lst4, lst5):
    lst3 = [value for value in lst1 if (value in lst2 and value in lst3 and value in lst4 and value in lst5)]
    return lst3

print("consider = ", consider)
print("intersection = ", intersection(consider[n.index(n[-2])], consider[n.index(n[-1])], consider[n.index(n[-3])], consider[n.index(n[-4])], consider[n.index(n[-5])]))