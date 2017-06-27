#include "kmeans.h"
#include <math_constants.h>
#include <iostream>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void clustering::Kmeans::do_saxpy(){
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

__global__
void gpu_assign_elements_2_clusters(const unsigned int n_e, const unsigned int n_c, const float * d_ex, const float * d_ey, const float * d_cx, const float * d_cy, std::size_t * d_c){
    for(std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_e; i += blockDim.x * gridDim.x){
        float best_cost = CUDART_INF_F;
        std::size_t best_cluster;
        for(std::size_t j = 0; j < n_c; ++j){
            float distance = fabsf(d_cx[j]-d_ex[i]) + fabsf(d_cy[j]-d_ey[i]);
            if(distance < best_cost){
                best_cost = distance;
                best_cluster = j;
            }
        }
    __syncthreads();
    d_c[i] = best_cluster;
    }
}

void gpu_print_device_info(){
    int dev_id;
    cudaDeviceProp prop;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&prop, dev_id);
    std::cout<<"Device Number: "<< dev_id <<std::endl;
    std::cout<<"Device name: "<< prop.name <<std::endl;
    std::cout<<"Memory Clock Rate (KHz): "<< prop.memoryClockRate <<std::endl;
    std::cout<<"Memory Bus Width (bits): "<< prop.memoryBusWidth <<std::endl;
    std::cout<<"Peak Memory Bandwidth (GB/s): "<< 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 <<std::endl;
    std::cout<<"MultiProcessor count: "<< prop.multiProcessorCount<<std::endl;
    std::cout<<"Max threads per MultiProcessor: "<< prop.maxThreadsPerMultiProcessor <<std::endl;
}

void gpu_update_clusters_centers(const unsigned int n_e, const unsigned int n_c, const float * h_ex, const float * h_ey, float * h_cx, float * h_cy, const std::size_t * h_c){
#pragma omp parallel for
    for(std::size_t i = 0; i < n_c; ++i){
        float x_c = 0, y_c = 0;
        unsigned int count = 0;
        for(std::size_t j = 0; j < n_e; j++){
            if(h_c[j] == i){
                ++count;
                x_c += h_ex[j];
                y_c += h_ey[j];
            }
        }
        if(count != 0){
            h_cx[i] = x_c/count;
            h_cy[i] = y_c/count;
        }
    }
}

void clustering::Kmeans::gpu_kmeans(unsigned int iterations, unsigned int n_blocks, unsigned int n_threads_per_block){
    float *h_ex, *h_ey, *h_cx, *h_cy,
          *d_ex, *d_ey, *d_cx, *d_cy;
    std::size_t *h_c, *d_c;
    unsigned int n_e = elements.size(), n_c = clusters.size();

    h_ex = (float*)malloc(n_e*sizeof(float));
    h_ey = (float*)malloc(n_e*sizeof(float));
    h_cx = (float*)malloc(n_c*sizeof(float));
    h_cy = (float*)malloc(n_c*sizeof(float));
    h_c = (std::size_t *)malloc(n_e*sizeof(std::size_t));

    for(std::size_t i = 0; i < n_e; ++i){
        h_ex[i] = elements.at(i).x();
        h_ey[i] = elements.at(i).y();
    }

    clear_all_clusters();

    for(std::size_t i = 0; i < n_c; ++i){
        h_cx[i] = clusters.at(i).x();
        h_cy[i] = clusters.at(i).y();
    }

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_ex, n_e*sizeof(float));
    cudaMalloc(&d_ey, n_e*sizeof(float));
    cudaMalloc(&d_cx, n_c*sizeof(float));
    cudaMalloc(&d_cy, n_c*sizeof(float));
    cudaMalloc(&d_c, n_e*sizeof(std::size_t));

    cudaMemcpy(d_ex, h_ex, n_e*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ey, h_ey, n_e*sizeof(float), cudaMemcpyHostToDevice);

    for(unsigned int i = 0; i < iterations; ++i){
        cudaMemcpy(d_cx, h_cx, n_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cy, h_cy, n_c*sizeof(float), cudaMemcpyHostToDevice);

        gpu_assign_elements_2_clusters<<<n_blocks, n_threads_per_block>>>(n_e, n_c, d_ex, d_ey, d_cx, d_cy, d_c);

        cudaMemcpy(h_c, d_c, n_e*sizeof(std::size_t), cudaMemcpyDeviceToHost);

        gpu_update_clusters_centers(n_e, n_c, h_ex, h_ey, h_cx, h_cy, h_c);
    }

    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms "<<std::endl;

    for(std::size_t i = 0; i < n_e; ++i){
        elements.at(i).cluster(h_c[i]);
        clusters.at(h_c[i]).insert_element(i);
    }

    for(std::size_t i = 0; i < n_c; ++i)
        clusters.at(i).update_center(h_cx[i], h_cy[i]);

    cudaFree(d_ex);
    cudaFree(d_ey);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_c);
    cudaFree(h_ex);
    cudaFree(h_ey);
    cudaFree(h_cx);
    cudaFree(h_cy);
    cudaFree(h_c);
}
