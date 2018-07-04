import matplotlib.pyplot as plt
import numpy as np

p5_rf1_path_length_matrix = np.loadtxt("p5/RF1_pl_matrix_lmbda1.txt")
p5_rf2_path_length_matrix = np.loadtxt("p5/RF2_pl_matrix_lmbda1.txt")
p5_rf3_path_length_matrix = np.loadtxt("p5/RF3_pl_matrix_lmbda1.txt")

p5_rf_path_length_matrix = np.zeros(p5_rf1_path_length_matrix.shape)

for i in range(p5_rf_path_length_matrix.shape[0]):
	for j in range(p5_rf_path_length_matrix.shape[1]):
		if(p5_rf1_path_length_matrix[i,j]==-1):
			p5_rf1_path_length_matrix[i,j] = 0
		if(p5_rf2_path_length_matrix[i,j]==-1):
			p5_rf2_path_length_matrix[i,j] = 0
		if(p5_rf3_path_length_matrix[i,j]==-1):
			p5_rf3_path_length_matrix[i,j] = 0

for i in range(p5_rf_path_length_matrix.shape[0]):
	for j in range(p5_rf_path_length_matrix.shape[1]):
		p5_rf_path_length_matrix[i,j] = (p5_rf1_path_length_matrix[i,j] + p5_rf2_path_length_matrix[i,j] + p5_rf3_path_length_matrix[i,j])/3.0

		if(p5_rf_path_length_matrix[i,j]<0.01):
			p5_rf_path_length_matrix[i,j] = -1


p7_rf1_path_length_matrix = np.loadtxt("p7/RF1_pl_matrix_lmbda1.txt")
p7_rf2_path_length_matrix = np.loadtxt("p7/RF2_pl_matrix_lmbda1.txt")
p7_rf3_path_length_matrix = np.loadtxt("p7/RF3_pl_matrix_lmbda1.txt")

p7_rf_path_length_matrix = np.zeros(p7_rf1_path_length_matrix.shape)

for i in range(p7_rf_path_length_matrix.shape[0]):
	for j in range(p7_rf_path_length_matrix.shape[1]):
		if(p7_rf1_path_length_matrix[i,j]==-1):
			p7_rf1_path_length_matrix[i,j] = 0
		if(p7_rf2_path_length_matrix[i,j]==-1):
			p7_rf2_path_length_matrix[i,j] = 0
		if(p7_rf3_path_length_matrix[i,j]==-1):
			p7_rf3_path_length_matrix[i,j] = 0

for i in range(p7_rf_path_length_matrix.shape[0]):
	for j in range(p7_rf_path_length_matrix.shape[1]):
		p7_rf_path_length_matrix[i,j] = (p7_rf1_path_length_matrix[i,j] + p7_rf2_path_length_matrix[i,j] + p7_rf3_path_length_matrix[i,j])/3.0

		if(p7_rf_path_length_matrix[i,j]<0.01):
			p7_rf_path_length_matrix[i,j] = -1

halton_path_length_matrix = np.loadtxt("Halton_pl_matrix_lmbda1.txt")

sp1_path_length_matrix = np.loadtxt("SP1_pl_matrix_lmbda1.txt")
sp2_path_length_matrix = np.loadtxt("SP2_pl_matrix_lmbda1.txt")
sp3_path_length_matrix = np.loadtxt("SP3_pl_matrix_lmbda1.txt")

sp_path_length_matrix = np.zeros(sp1_path_length_matrix.shape)
for i in range(sp_path_length_matrix.shape[0]):
	for j in range(sp_path_length_matrix.shape[1]):
		if(sp1_path_length_matrix[i,j]==-1):
			sp1_path_length_matrix[i,j] = 0
		if(sp2_path_length_matrix[i,j]==-1):
			sp2_path_length_matrix[i,j] = 0
		if(sp3_path_length_matrix[i,j]==-1):
			sp3_path_length_matrix[i,j] = 0

for i in range(sp_path_length_matrix.shape[0]):
	for j in range(sp_path_length_matrix.shape[1]):
		sp_path_length_matrix[i,j] = (sp1_path_length_matrix[i,j] + sp2_path_length_matrix[i,j] + sp3_path_length_matrix[i,j])/3.0

		if(sp_path_length_matrix[i,j]<0.01):
			sp_path_length_matrix[i,j] = -1

n = [200, 400, 600, 800, 1000, 1200]

sr_p5_rf = [0, 0, 0, 0, 0, 0]
sr_p7_rf = [0, 0, 0, 0, 0, 0]
sr_halton = [0, 0, 0, 0, 0, 0]
sr_sp = [0, 0, 0, 0, 0, 0,]

hfont = {'fontname': 'Helvetica'}

for i in range(len(n)):
	for j in range(len(p5_rf_path_length_matrix[i])):
		path_length_p5_rf = p5_rf_path_length_matrix[i,j]
		if(not path_length_p5_rf==-1):
			sr_p5_rf[i] += 1

		path_length_p7_rf = p7_rf_path_length_matrix[i,j]
		if(not path_length_p7_rf==-1):
			sr_p7_rf[i] += 1

		path_length_halton = halton_path_length_matrix[i,j]
		if(not path_length_halton==-1):
			sr_halton[i] += 1

		path_length_sp = sp_path_length_matrix[i,j]
		if(not path_length_sp==-1):
			sr_sp[i] += 1

# plt.plot(n, sr_p5_rf, color = "green", linewidth = 2, label = "RF+Halton(50:50)")
plt.plot(n, sr_p7_rf, color = "orange", linewidth = 2, label = "RF+Halton(30:70)")
plt.plot(n, sr_sp, color = "blue", linewidth = 2, label = "SP+Halton(50:50)")
plt.plot(n, sr_halton, color = "red", linewidth = 2, label = "Halton")

plt.xlabel("No of Samples ", **hfont)
plt.ylabel("Success % ", **hfont)

leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize='medium')

plt.xlim(0, 2000)
plt.ylim(0, 110)
plt.title("Hard", **hfont)
plt.savefig("Success_Rate.jpg", bbox_inches='tight')
plt.show()