import matplotlib.pyplot as plt
import numpy as np

# p5_rf_h_matrix = np.loadtxt("p5/RF_h_1_pl_matrix_lmbda1.txt")

# p7_rf_e_matrix = np.loadtxt("RF_e_pl_matrix_lmbda1.txt")
p7_rf_m_matrix = np.loadtxt("p7/RF_m_1_pl_matrix_lmbda1.txt")
# p7_rf_h_matrix = np.loadtxt("p7/RF_h_3_pl_matrix_lmbda1.txt")

# sp_e_matrix = np.loadtxt("SP_e_pl_matrix_lmbda1.txt")
sp_m_matrix = np.loadtxt("SP/SP_m_1_pl_matrix_lmbda1.txt")
# sp_h_matrix = np.loadtxt("SP/SP_h_1_pl_matrix_lmbda1.txt")

# halton_e_matrix = np.loadtxt("Halton_pl_e_matrix_lmbda1.txt")
halton_m_matrix = np.loadtxt("Halton/Halton_pl_h_matrix_lmbda1.txt")
# halton_h_matrix = np.loadtxt("Halton_pl_h_matrix_lmbda1.txt")

n = [200, 400, 600, 800, 1000, 1200, 1500, 2000]


sr_p5_e_rf = [0, 0, 0, 0, 0, 0, 0, 0]
sr_p5_m_rf = [0, 0, 0, 0, 0, 0, 0, 0]
sr_p5_h_rf = [0, 0, 0, 0, 0, 0, 0, 0]

sr_p7_e_rf = [0, 0, 0, 0, 0, 0, 0, 0]
sr_p7_m_rf = [0, 0, 0, 0, 0, 0, 0, 0]
sr_p7_h_rf = [0, 0, 0, 0, 0, 0, 0, 0]

sr_e_halton = [0, 0, 0, 0, 0, 0, 0, 0]
sr_m_halton = [0, 0, 0, 0, 0, 0, 0, 0]
sr_h_halton = [0, 0, 0, 0, 0, 0, 0, 0]

sr_e_sp = [0, 0, 0, 0, 0, 0, 0, 0]
sr_m_sp = [0, 0, 0, 0, 0, 0, 0, 0]
sr_h_sp = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(n)):
	for j in range(len(p7_rf_m_matrix[i])):

		# if(not p5_rf_m_matrix[i,j]==-1):
		# 	sr_p5_m_rf[i] += 1

		# if(not p7_rf_e_matrix[i,j]==-1):
		# 	sr_p7_e_rf[i] += 1
		# if(not p7_rf_m_matrix[i,j]==-1):
		# 	sr_p7_m_rf[i] += 1
		if(not p7_rf_m_matrix[i,j]==-1):
			sr_p7_m_rf[i] += 1


		# if(not halton_e_matrix[i,j]==-1):
		# 	sr_e_halton[i] += 1
		# if(not halton_m_matrix[i,j]==-1):
		# 	sr_m_halton[i] += 1
		if(not halton_m_matrix[i,j]==-1):
			sr_m_halton[i] += 1

		# if(not sp_e_matrix[i,j]==-1):
		# 	sr_e_sp[i] += 1
		# if(not sp_m_matrix[i,j]==-1):
		# 	sr_m_sp[i] += 1
		if(not sp_m_matrix[i,j]==-1):
			sr_m_sp[i] += 1

hfont = {'fontname': 'Helvetica'}
plt.xlabel("No of Samples ", **hfont)
plt.ylabel("Success % ", **hfont)

t = "_m_"

plt.plot(n, sr_p7_m_rf, color = "orange", linewidth = 2, label = "RF+Halton(30:70)")
plt.plot(n, sr_p5_m_rf, color = "green", linewidth = 2, label = "RF+Halton(50:50)")
plt.plot(n, sr_m_halton, color = "red", linewidth = 2, label = "Halton")
plt.plot(n, sr_m_sp, color = "blue", linewidth = 2, label = "SP+Halton(50:50)")

leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize='medium')

plt.xlim(0, 4000)
plt.ylim(0, 110)
plt.title(t, **hfont)
# plt.savefig("Medium.png", bbox_inches='tight')
plt.show()