home:
	nvcc Main.cu -o erosion -O3 -Iexternal

server:
	/usr/local/cuda-7.5/bin/nvcc Main.cu -o erosion -O3 -Iexternal