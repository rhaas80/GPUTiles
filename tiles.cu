#include <cuda.h>
#include <iostream>
#include <cstdlib>

#define TILE_SX 32
#define TILE_SY 32

double __device__ dx(const double * v) {
    return 0.5*(v[+1] - v[-1]);
}

double __device__ dy(const double * v, const int dj) {
    return 0.5*(v[dj] - v[-dj]);
}

#define CUDA_CHECK do { \
  cudaError res = cudaGetLastError(); \
  if(res != cudaSuccess) { \
    std::cerr << "CUDA Failure at " << __LINE__ << " " << cudaGetErrorString(res) << "\n"; \
    exit(1); \
  } \
} while(0) 

void __global__ derivs(const double * v, double *v_x, double *v_y, double *v_xy) {
  double __shared__ tile[TILE_SX * TILE_SY];
  double __shared__ tile_x[TILE_SX * TILE_SY];

  int idx = threadIdx.x + threadIdx.y*TILE_SX;
  tile[idx] = v[idx]; 
  __syncthreads();
  
  if(threadIdx.x > 0 && threadIdx.x < TILE_SX - 1) {
    v_x[idx] = tile_x[idx] = dx(&tile[idx]);
  }

  __syncthreads();

  if(threadIdx.y > 0 && threadIdx.y < TILE_SX - 1) {
    v_y[idx] = dy(&tile[idx], TILE_SY);
    v_xy[idx] = dy(&tile_x[idx], TILE_SY);
  }
}

int main(void) {
  double *v = (double*)calloc(TILE_SX * TILE_SY, sizeof(*v));
  double *v_x = (double*)calloc(TILE_SX * TILE_SY, sizeof(*v_x));
  double *v_y = (double*)calloc(TILE_SX * TILE_SY, sizeof(*v_y));
  double *v_xy = (double*)calloc(TILE_SX * TILE_SY, sizeof(*v_xy));
  double *d_v, *d_v_x, *d_v_y, *d_v_xy;
  cudaMalloc(&d_v, TILE_SX * TILE_SY * sizeof(*v));
  CUDA_CHECK;
  cudaMalloc(&d_v_x, TILE_SX * TILE_SY * sizeof(*v_x));
  CUDA_CHECK;
  cudaMalloc(&d_v_y, TILE_SX * TILE_SY * sizeof(*v_y));
  CUDA_CHECK;
  cudaMalloc(&d_v_xy, TILE_SX * TILE_SY * sizeof(*v_xy));
  CUDA_CHECK;

  cudaMemcpy(d_v, v, TILE_SX * TILE_SY * sizeof(*v), cudaMemcpyHostToDevice);
  CUDA_CHECK;
  dim3 dimBlock(TILE_SX, TILE_SY);
  dim3 dimGrid(1, 1);
  derivs<<<dimGrid, dimBlock>>>(d_v, d_v_x, d_v_y, d_v_xy);
  CUDA_CHECK;
  cudaMemcpy(d_v_x, v_x, TILE_SX * TILE_SY * sizeof(*v_x), cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(d_v_y, v_y, TILE_SX * TILE_SY * sizeof(*v_y), cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(d_v_xy, v_xy, TILE_SX * TILE_SY * sizeof(*v_xy), cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  return 0;
}
