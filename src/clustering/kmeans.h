#ifndef KMEANS_KMEANS_H
#define KMEANS_KMEANS_H

#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace clustering{

class Element{
private:
    float pos_x, pos_y;
    std::size_t cluster_best;
public:
    Element(float pos_x, float pos_y):pos_x(pos_x), pos_y(pos_y){}
    float x() const {return pos_x;}
    float y() const {return pos_y;}
    std::size_t cluster() const {return cluster_best;}
    void cluster(std::size_t cluster_id){cluster_best = cluster_id;}
};

class Cluster{
private:
    float pos_x, pos_y;
    std::vector<std::size_t> elements;
public:
    Cluster(float pos_x, float pos_y):pos_x(pos_x), pos_y(pos_y){}
    float x() const {return pos_x;}
    float y() const {return pos_y;}
    const std::vector<std::size_t> & cluster_elements() const {return elements;}
    void reserve(std::size_t size){elements.reserve(size);}
    void clear(){elements.clear();}
    void insert_element(std::size_t element_id){elements.push_back(element_id);}
    void update_center(float x, float y){pos_x = x; pos_y = y;}
};

class Kmeans{
private:
    unsigned int max_cluster_size;
    std::vector<Element> elements;
    std::vector<Cluster> clusters;
    void update_clusters_centers();
    void p_update_clusters_centers();
    void clear_all_clusters(){for(auto & c : clusters)c.clear();}
    void assign_elements_2_clusters();
    void p_assign_elements_2_clusters();
    void resolve_overflows();
    void clear_empty_clusters();
public:
    Kmeans(){}
    const Cluster & cluster(std::size_t i) const {return clusters.at(i);}
    const Element & element(std::size_t i) const {return elements.at(i);}
    const std::vector<Element> & k_elements() const {return elements;}
    const std::vector<Cluster> & k_clusters() const {return clusters;}
    void set_max_cluster_size(unsigned int max_size){max_cluster_size = max_size;}
    void reserve_clusters(std::size_t size){clusters.reserve(size);}
    void reserve_elements(std::size_t size){elements.reserve(size);}
    void add_cluster(float pos_x, float pos_y){clusters.push_back(Cluster(pos_x, pos_y));}
    void add_element(float pos_x, float pos_y){elements.push_back(Element(pos_x, pos_y));}
    void kmeans(unsigned int iterations);
    void p_kmeans(unsigned int iterations);
    void gpu_kmeans(unsigned int iterations, unsigned int n_blocks, unsigned int n_threads_per_block);
    void do_saxpy();
};

}

#endif // KMEANS_KMEANS_H
