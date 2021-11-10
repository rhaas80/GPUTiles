#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#define TILE_SX 32
#define TILE_SY 32
#define TILES_X 10
#define TILES_Y 10

float __device__ dx(const float * v) {
    return 0.5*(v[+1] - v[-1]);
}

float __device__ dy(const float * v, const int dj) {
    return 0.5*(v[dj] - v[-dj]);
}

float __device__ dxy(const float * v, const int dj) {
    return 0.25*((v[dj+1]-v[dj-1]) - (v[-dj+1] - v[-dj-1]));
}

#define CUDA_CHECK do { \
  cudaError res = cudaGetLastError(); \
  if(res != cudaSuccess) { \
    std::cerr << "CUDA Failure at " << __LINE__ << " " << cudaGetErrorString(res) << "\n"; \
    exit(1); \
  } \
} while(0) 

void __global__ derivs(const float * v, float *v_x, float *v_y, float *v_xy) {
  float __shared__ tile[TILE_SX * TILE_SY];
  float __shared__ tile_x[TILE_SX * TILE_SY];

  int tidx = threadIdx.x + threadIdx.y*TILE_SX;
  int idx = threadIdx.x + threadIdx.y*TILE_SX + blockIdx.x*TILE_SX*TILE_SY 
            + blockIdx.y*TILES_X*TILE_SX*TILE_SY;
  tile[tidx] = v[idx]; 
  __syncthreads();
  
  if(threadIdx.x > 0 && threadIdx.x < TILE_SX - 1) {
    v_x[idx] = tile_x[tidx] = dx(&tile[tidx]);
  }

  __syncthreads();

  if(threadIdx.y > 0 && threadIdx.y < TILE_SY - 1) {
    v_y[idx] = dy(&tile[tidx], TILE_SY);
    v_xy[idx] = dy(&tile_x[tidx], TILE_SY);
  }
}

void __global__ derivs_naive(const float * v, float *v_x, float *v_y, float *v_xy) {
  int idx = threadIdx.x + threadIdx.y*TILE_SX + blockIdx.x*TILE_SX*TILE_SY 
            + blockIdx.y*TILES_X*TILE_SX*TILE_SY;
  
  if(threadIdx.x > 0 && threadIdx.x < TILE_SX - 1 &&
     threadIdx.y > 0 && threadIdx.y < TILE_SY - 1) {
    v_x[idx] = dx(&v[idx]);
    v_y[idx] = dy(&v[idx], TILE_SY);
    v_xy[idx] = dxy(&v[idx], TILE_SY);
  }
}

float now() {
  timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec*1e-6;
}

int main(void) {
  float *v = (float*)calloc(TILE_SX * TILE_SY * TILES_X * TILES_Y, sizeof(*v));
  float *v_x = (float*)calloc(TILE_SX * TILE_SY * TILES_X * TILES_Y, sizeof(*v_x));
  float *v_y = (float*)calloc(TILE_SX * TILE_SY * TILES_X * TILES_Y, sizeof(*v_y));
  float *v_xy = (float*)calloc(TILE_SX * TILE_SY * TILES_X * TILES_Y, sizeof(*v_xy));
  float *d_v, *d_v_x, *d_v_y, *d_v_xy;
  cudaMalloc(&d_v, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v));
  CUDA_CHECK;
  cudaMalloc(&d_v_x, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_x));
  CUDA_CHECK;
  cudaMalloc(&d_v_y, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_y));
  CUDA_CHECK;
  cudaMalloc(&d_v_xy, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_xy));
  CUDA_CHECK;

  // disable caches for testing
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  for(int j=0;j<2;j++) {
  {
    cudaMemcpy(d_v, v, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    dim3 dimBlock(TILE_SX, TILE_SY, 1);
    dim3 dimGrid(TILES_X, TILES_Y, 1);
  double start = now();
  // TODO: run one without timint to get rid of initialization ost
  for(int i = 0 ; i < 1000 ; i++) {
    derivs<<<dimGrid, dimBlock>>>(d_v, d_v_x, d_v_y, d_v_xy);
  }
  double end = now();
  std::cout << "tiled took " << (end-start) << " s\n";
    CUDA_CHECK;
    cudaMemcpy(v_x, d_v_x, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_x), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(v_y, d_v_y, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_y), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(v_xy, d_v_xy, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_xy), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
  }
  {
    cudaMemcpy(d_v, v, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v), cudaMemcpyHostToDevice);
    CUDA_CHECK;
    dim3 dimBlock(TILE_SX, TILE_SY);
    dim3 dimGrid(TILES_X, TILES_Y);
  double start = now();
  // TODO: run one without timint to get rid of initialization ost
  for(int i = 0 ; i < 1000 ; i++) {
    derivs_naive<<<dimGrid, dimBlock>>>(d_v, d_v_x, d_v_y, d_v_xy);
  }
  double end = now();
  std::cout << "naive took " << (end-start) << " s\n";
    CUDA_CHECK;
    cudaMemcpy(v_x, d_v_x, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_x), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(v_y, d_v_y, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_y), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    cudaMemcpy(v_xy, d_v_xy, TILE_SX * TILE_SY * TILES_X * TILES_Y * sizeof(*v_xy), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
  }
  }

  return 0;
}
