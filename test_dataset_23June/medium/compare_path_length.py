import numpy as np

def main():
	numviz = 50
	halton_p_addr = "./path_lengths_halton_numviz"+str(numviz*2)+"_k7.txt"
	RF_p_addr = "./path_lengths_RF_numviz"+str(numviz)+"_k7.txt"
	SP_p_addr = "./path_lengths_SP_numviz"+str(numviz)+"_k7.txt"

	halton_path_lengths = np.loadtxt(halton_p_addr)
	RF_path_lengths = np.loadtxt(RF_p_addr)
	SP_path_lengths = np.loadtxt(SP_p_addr)

	n = 0.0
	halton = SP = RF = 0

	print("l = ", len(RF_path_lengths))

	for i in range(len(RF_path_lengths)):
		if(halton_path_lengths[i]==-1 or RF_path_lengths[i]==-1 or SP_path_lengths[i]==-1):
			# print("i = ", i)
			continue
		else:
			n += 1
			# print("i1 = ",i)
			halton += halton_path_lengths[i]
			SP += SP_path_lengths[i]
			RF += RF_path_lengths[i]

	print("Avg path lengths: num_viz = ", numviz, " n = ", n)
	print("halton: ", halton/n)
	print("SP: ", SP/n)
	print("RF: ", RF/n)

if __name__ == '__main__':
	main()