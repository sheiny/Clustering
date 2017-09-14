#include "kmeans.h"

namespace clustering{
void Kmeans::kmeans(unsigned int iterations){
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    time_start = std::chrono::high_resolution_clock::now();
    update_rtree();
    for(unsigned int i = 1; i <= iterations; ++i){
        clear_all_clusters();
        assign_elements_2_clusters();
        update_clusters_centers();
        clear_empty_clusters();
        resolve_overflows();
        update_rtree();
    }
    time_end = std::chrono::high_resolution_clock::now();
    auto total_time = time_end - time_start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count()<<" ms ";
}

void Kmeans::update_rtree(){
    clusters_rtree.clear();
    for(std::size_t c_index = 0; c_index < c_positions.size(); c_index++)
        clusters_rtree.insert(rtree_node(c_positions.at(c_index), c_index));
}

void Kmeans::update_clusters_centers(){
    for(std::size_t c_index = 0; c_index < c_positions.size(); c_index++){
        double x = 0, y = 0;
        for(auto e_position : c_elements.at(c_index)){
            x += e_position.x();
            y += e_position.y();
        }
        x = x/c_elements.at(c_index).size();
        y = y/c_elements.at(c_index).size();
        c_positions.at(c_index) = point(x, y);
    }
}

void Kmeans::assign_elements_2_clusters(){
    for(std::size_t e_index = 0; e_index < e_positions.size(); e_index++){
        std::vector<rtree_node> closest_nodes;
        clusters_rtree.query(boost::geometry::index::nearest(e_positions.at(e_index), 1), std::back_inserter(closest_nodes));
        auto closest_cluster = closest_nodes.front();
        c_elements.at(closest_cluster.second).push_back(e_positions.at(e_index));
    }
}

void Kmeans::clear_empty_clusters(){
    for(int c_index = c_elements.size()-1; c_index >= 0; c_index--)
        if(c_elements.at(c_index).empty()){
            c_positions.erase(c_positions.begin()+c_index);
            c_elements.erase(c_elements.begin()+c_index);
        }
}

void Kmeans::resolve_overflows(){
    for(int c_index = c_elements.size()-1; c_index >= 0; c_index--)
        if(c_elements.at(c_index).size() > max_cluster_size){
            std::vector<point> elements_overflow = c_elements.at(c_index);
            c_elements.erase(c_elements.begin()+c_index);
            for(std::size_t i = 0; i < elements_overflow.size(); i++){
                if(i % max_cluster_size == 0)
                    c_elements.push_back(std::vector<point>());
                c_elements.back().push_back(elements_overflow.at(i));
            }
        }
    c_positions.resize(c_elements.size());
}
}
