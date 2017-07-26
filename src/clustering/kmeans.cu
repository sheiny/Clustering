#include "kmeans.h"
#include <math_constants.h>
#include <iostream>
#include <unordered_map>

__global__
void saxpy(int n, float a, float *x, float *y){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void clustering::Kmeans::do_saxpy(){
  int N = 1<<20;
  std::vector<float> x, y;
  x.resize(N);
  y.resize(N);

  float *d_x, *d_y;
  
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));
  
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  
  cudaMemcpy(d_x, &x[0], N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &y[0], N*sizeof(float), cudaMemcpyHostToDevice);
  
  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(&y[0], d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  float maxError = 0.0f; 
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  
  printf("Max error: %f\n", maxError);
  cudaFree(d_x);
  cudaFree(d_y);
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

void gpu_update_clusters_centers(std::vector<float> & h_ex, std::vector<float> & h_ey, std::vector<float> & h_cx, std::vector<float> & h_cy, std::vector<std::size_t> & h_c){
#pragma omp parallel for
    for(std::size_t i = 0; i < h_cx.size(); ++i){
        float x_c = 0, y_c = 0;
        unsigned int count = 0;
        for(std::size_t j = 0; j < h_ex.size(); j++){
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

void gpu_clear_empty_clusters(std::vector<std::size_t> & h_c, std::vector<float> & h_cx, std::vector<float> & h_cy){
    std::unordered_map<std::size_t, std::size_t> new_assignment;
    std::size_t new_cluster_index = 0;
    for(auto & assigned_cluster : h_c){
        auto search = new_assignment.find(assigned_cluster);
        if(search == new_assignment.end()){
            new_assignment.insert({assigned_cluster, new_cluster_index});
            assigned_cluster = new_cluster_index;
            ++new_cluster_index;
        }else
            assigned_cluster = search->second;
    }
    h_cx.resize(new_assignment.size());
    h_cy.resize(new_assignment.size());
}

void gpu_resolve_overflows(std::vector<std::size_t> & h_c, std::vector<float> & h_cx, std::vector<float> & h_cy, unsigned int max_cluster_size){
    std::vector<std::pair<unsigned int, std::size_t>> occurrences;
    occurrences.resize(h_cx.size(), std::pair<unsigned int, std::size_t>(0, std::numeric_limits<std::size_t>::max()));
    for(auto & assigned_cluster: h_c){
        if(occurrences.at(assigned_cluster).first < max_cluster_size)
            ++occurrences.at(assigned_cluster).first;
        else{
            std::size_t new_cluster;
            if(occurrences.at(assigned_cluster).second != std::numeric_limits<std::size_t>::max())
                new_cluster = occurrences.at(assigned_cluster).second;
            else{
                occurrences.push_back(std::pair<unsigned int, std::size_t>(0, std::numeric_limits<std::size_t>::max()));
                new_cluster = occurrences.size()-1;
                occurrences.at(assigned_cluster).second = new_cluster;
            }
            ++occurrences.at(new_cluster).first;
            if(occurrences.at(new_cluster).first == max_cluster_size);
                occurrences.at(assigned_cluster).second = std::numeric_limits<std::size_t>::max();
            assigned_cluster = new_cluster;
        }
    }
    h_cx.resize(occurrences.size());
    h_cy.resize(occurrences.size());
}

void clustering::Kmeans::gpu_kmeans(unsigned int iterations, unsigned int n_blocks, unsigned int n_threads_per_block){
    float *d_ex, *d_ey, *d_cx, *d_cy;
    std::vector<float> h_ex, h_ey, h_cx, h_cy;
    std::size_t *d_c;
    std::vector<std::size_t> h_c;

    h_ex.reserve(elements.size());
    h_ey.reserve(elements.size());
    h_cx.reserve(clusters.size());
    h_cy.reserve(clusters.size());
    h_c.resize(elements.size());

    for(std::size_t i = 0; i < elements.size(); ++i){
        h_ex.push_back(elements.at(i).x());
        h_ey.push_back(elements.at(i).y());
    }
    for(std::size_t i = 0; i < clusters.size(); ++i){
        h_cx.push_back(clusters.at(i).x());
        h_cy.push_back(clusters.at(i).y());
    }

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_ex, h_ex.size()*sizeof(float));
    cudaMalloc(&d_ey, h_ey.size()*sizeof(float));
    cudaMalloc(&d_c, elements.size()*sizeof(std::size_t));
    cudaMemcpy(d_ex, &h_ex[0], h_ex.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ey, &h_ey[0], h_ey.size()*sizeof(float), cudaMemcpyHostToDevice);

    for(unsigned int i = 1; i <= iterations; ++i){
        cudaMalloc(&d_cx, h_cx.size()*sizeof(float));
        cudaMalloc(&d_cy, h_cy.size()*sizeof(float));
        cudaMemcpy(d_cx, &h_cx[0], h_cx.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cy, &h_cy[0], h_cy.size()*sizeof(float), cudaMemcpyHostToDevice);

        gpu_assign_elements_2_clusters<<<n_blocks, n_threads_per_block>>>(elements.size(), h_cx.size(), d_ex, d_ey, d_cx, d_cy, d_c);

        cudaMemcpy(&h_c[0], d_c, elements.size()*sizeof(std::size_t), cudaMemcpyDeviceToHost);

        if(i % 5 == 0){
            gpu_clear_empty_clusters(h_c, h_cx, h_cy);
            gpu_resolve_overflows(h_c, h_cx, h_cy, max_cluster_size);
        }

        gpu_update_clusters_centers(h_ex, h_ey, h_cx, h_cy, h_c);
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms "<<std::endl;

    clusters.clear();
    clusters.reserve(h_cx.size());
    for(std::size_t i = 0; i < h_cx.size(); ++i)
        add_cluster(h_cx[i], h_cy[i]);

    for(std::size_t i = 0; i < elements.size(); ++i){
        elements.at(i).cluster(h_c[i]);
        clusters.at(h_c[i]).insert_element(i);
    }
 
    cudaFree(d_ex);
    cudaFree(d_ey);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_c);
}
