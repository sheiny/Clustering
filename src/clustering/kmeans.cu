#include "kmeans.h"
#include <math_constants.h>

__global__
void gpu_assign_elements_2_clusters(unsigned int n_e, unsigned int n_c, float * d_eposx, float * d_eposy, float * d_cposx, float * d_cposy, unsigned int * d_cbests){
    for(std::size_t i = 0; i < n_e; ++i){
        float best_cost = CUDART_INF_F;
        for(std::size_t j = 0; j < n_c; ++j){
            float distance = fabsf(d_cposx[j]-d_eposx[i]) + fabsf(d_cposy[j]-d_eposy[i]);
            if(distance < best_cost){
                best_cost = distance;
                d_cbests[i] = j;
            }
        }
    }
}

void clustering::Kmeans::kmeans_gpu(unsigned int iterations){
    float *h_eposx, *h_eposy, *h_cposx, *h_cposy,
          *d_eposx, *d_eposy, *d_cposx, *d_cposy;
    unsigned int *h_cbests, *d_cbests, n_e = elements.size(), n_c = clusters.size();

    clear_all_clusters();

    h_eposx = (float*)malloc(n_e*sizeof(float));
    h_eposy = (float*)malloc(n_e*sizeof(float));
    h_cposx = (float*)malloc(n_c*sizeof(float));
    h_cposy = (float*)malloc(n_c*sizeof(float));
    h_cbests = (unsigned int*)malloc(n_e*sizeof(unsigned int));

    for(std::size_t i = 0; i < n_e; ++i){
        h_eposx[i] = elements.at(i).x();
        h_eposy[i] = elements.at(i).y();
        h_cbests[i] = std::numeric_limits<unsigned int>::max();
    }

    for(std::size_t i = 0; i < n_c; ++i){
        h_cposx[i] = clusters.at(i).x();
        h_cposy[i] = clusters.at(i).y();
    }

    cudaMalloc(&d_eposx, n_e*sizeof(float));
    cudaMalloc(&d_eposy, n_e*sizeof(float));
    cudaMalloc(&d_cposx, n_c*sizeof(float));
    cudaMalloc(&d_cposy, n_c*sizeof(float));
    cudaMalloc(&d_cbests, n_e*sizeof(unsigned int));

    cudaMemcpy(d_eposx, h_eposx, n_e*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eposy, h_eposy, n_e*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cposx, h_cposx, n_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cposy, h_cposy, n_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cbests, h_cbests, n_e*sizeof(unsigned int), cudaMemcpyHostToDevice);

    gpu_assign_elements_2_clusters<<<1, 1>>>(n_e, n_c, d_eposx, d_eposy, d_cposx, d_cposy, d_cbests);

    cudaMemcpy(h_cbests, d_cbests, n_e*sizeof(float), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < elements.size(); ++i){
        elements.at(i).cluster(h_cbests[i]);
        clusters.at(h_cbests[i]).insert_element(i);
    }

    update_clusters_centers();

    cudaFree(d_eposx);
    cudaFree(d_eposy);
    cudaFree(d_cposx);
    cudaFree(d_cposy);
    cudaFree(d_cbests);
}
