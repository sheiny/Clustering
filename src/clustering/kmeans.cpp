#include "kmeans.h"

void clustering::Kmeans::kmeans(unsigned int iterations){
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 0; i < iterations; ++i){
        clear_all_clusters();
        assign_elements_2_clusters();
        update_clusters_centers();
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms "<<std::endl;
}

void clustering::Kmeans::p_kmeans(unsigned int iterations){
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 0; i < iterations; ++i){
        clear_all_clusters();
        p_assign_elements_2_clusters();
        for(std::size_t i = 0; i < elements.size(); ++i)
            clusters.at(elements.at(i).cluster()).insert_element(i);
        p_update_clusters_centers();
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms "<<std::endl;
}

void clustering::Kmeans::update_clusters_centers(){
    for(auto & cluster : clusters){
        if(!cluster.cluster_elements().empty()){
            float x_c = 0, y_c = 0;
            for(auto element_id : cluster.cluster_elements()){
                x_c += elements.at(element_id).x();
                y_c += elements.at(element_id).y();
            }
            x_c = x_c / cluster.cluster_elements().size();
            y_c = y_c / cluster.cluster_elements().size();
            cluster.update_center(x_c, y_c);
        }
    }
}

void clustering::Kmeans::p_update_clusters_centers(){
#pragma omp parallel for
    for(auto cluster_it = clusters.begin(); cluster_it < clusters.end(); ++cluster_it){
        if(!cluster_it->cluster_elements().empty()){
            float x_c = 0, y_c = 0;
            for(auto element_id : cluster_it->cluster_elements()){
                x_c += elements.at(element_id).x();
                y_c += elements.at(element_id).y();
            }
            x_c = x_c / cluster_it->cluster_elements().size();
            y_c = y_c / cluster_it->cluster_elements().size();
            cluster_it->update_center(x_c, y_c);
        }
    }
}

void clustering::Kmeans::assign_elements_2_clusters(){
    for(auto element_it = elements.begin(); element_it != elements.end(); ++element_it){
        element_it->cluster(std::numeric_limits<std::size_t>::max());
        float best_cost = std::numeric_limits<float>::max();
        for(auto cluster_it = clusters.begin(); cluster_it != clusters.end(); ++cluster_it){
            float distance = std::abs(cluster_it->x() - element_it->x()) + std::abs(cluster_it->y() - element_it->y());
            if(distance < best_cost){
                best_cost = distance;
                element_it->cluster(std::distance(clusters.begin(), cluster_it));
            }
        }
        clusters.at(element_it->cluster()).insert_element(std::distance(elements.begin(), element_it));
    }
}

void clustering::Kmeans::p_assign_elements_2_clusters(){
#pragma omp parallel for
    for(auto element_it = elements.begin(); element_it < elements.end(); ++element_it){
        element_it->cluster(std::numeric_limits<std::size_t>::max());
        float best_cost = std::numeric_limits<float>::max();
        for(auto cluster_it = clusters.begin(); cluster_it != clusters.end(); ++cluster_it){
            float distance = std::abs(cluster_it->x() - element_it->x()) + std::abs(cluster_it->y() - element_it->y());
            if(distance < best_cost){
                best_cost = distance;
                element_it->cluster(std::distance(clusters.begin(), cluster_it));
            }
        }
    }
}
