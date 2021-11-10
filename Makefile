CFLAGS = -Wall
tiles: tiles.cu
	nvcc -o $@ $^
