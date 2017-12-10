all:
	nvcc Main.cu -o erosion -O3 -Iexternal