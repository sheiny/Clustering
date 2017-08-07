#include "kmeans.h"

namespace clustering{
void Kmeans::kmeans(unsigned int iterations){
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 1; i <= iterations; ++i){
        clear_all_clusters();
        assign_elements_2_clusters();
        update_clusters_centers();
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms ";
    clear_empty_clusters();
    resolve_overflows();

}

void Kmeans::p_kmeans(unsigned int iterations){
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 1; i <= iterations; ++i){
        clear_all_clusters();
        p_assign_elements_2_clusters();
        p_update_clusters_centers();
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms ";
    clear_empty_clusters();
    resolve_overflows();

}

void Kmeans::update_clusters_centers(){
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

void Kmeans::p_update_clusters_centers(){
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

void Kmeans::assign_elements_2_clusters(){
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

void Kmeans::p_assign_elements_2_clusters(){
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
    for(std::size_t i = 0; i < elements.size(); ++i)
        clusters.at(elements.at(i).cluster()).insert_element(i);
}

void Kmeans::resolve_overflows(){
    std::vector<Cluster> new_clusters;
    for(auto & cluster : clusters)
        if(cluster.cluster_elements().size() > max_cluster_size){
            auto cluster_elements = cluster.cluster_elements();
            new_clusters.insert(new_clusters.end(), std::ceil(cluster_elements.size()/(float)max_cluster_size), Cluster(cluster.x(), cluster.y()));
            for(auto element_it = cluster_elements.begin(); element_it != cluster_elements.end(); element_it++)
                new_clusters.at(new_clusters.size()-1-std::floor(std::distance(cluster_elements.begin(), element_it)/(float)max_cluster_size)).insert_element(*element_it);//improve this
            cluster = new_clusters.back();
            new_clusters.pop_back();
        }
    for(auto & new_cluster : new_clusters)
        clusters.push_back(new_cluster);
}

bool empty_cluster(const clustering::Cluster & c){
    return c.cluster_elements().empty();
}

void Kmeans::clear_empty_clusters(){
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(), empty_cluster), clusters.end());
}

}
