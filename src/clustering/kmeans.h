#ifndef KMEANS_KMEANS_H
#define KMEANS_KMEANS_H

#include <vector>
#include <limits>
#include <cmath>
#include <omp.h>

namespace clustering{

class Element{
private:
    float pos_x, pos_y;
    unsigned int cluster_best;
public:
    Element(float pos_x, float pos_y):pos_x(pos_x), pos_y(pos_y){}
    float x() const {return pos_x;}
    float y() const {return pos_y;}
    unsigned int cluster() const {return cluster_best;}
    void cluster(unsigned int cluster_id){cluster_best = cluster_id;}
};

class Cluster{
private:
    float pos_x, pos_y;
    std::vector<unsigned int> elements;
public:
    Cluster(float pos_x, float pos_y):pos_x(pos_x), pos_y(pos_y){}
    float x() const {return pos_x;}
    float y() const {return pos_y;}
    const std::vector<unsigned int> & cluster_elements() const {return elements;}
    void clear(){elements.clear();}
    void insert_element(unsigned int element_id){elements.push_back(element_id);}
    void update_center(float x, float y){pos_x = x; pos_y = y;}
};

class Kmeans{
private:
    std::vector<Element> elements;
    std::vector<Cluster> clusters;
    void update_clusters_centers();
    void p_update_clusters_centers();
    void clear_all_clusters(){for(auto & c : clusters)c.clear();}
    void assign_elements_2_clusters();
    void p_assign_elements_2_clusters();
public:
    Kmeans(){}
    const Cluster & cluster(unsigned int i) const {return clusters.at(i);}
    const Element & element(unsigned int i) const {return elements.at(i);}
    const std::vector<Element> & k_elements() const {return elements;}
    const std::vector<Cluster> & k_clusters() const {return clusters;}
    void reserve_clusters(std::size_t size){clusters.reserve(size);}
    void reserve_elements(std::size_t size){elements.reserve(size);}
    void add_cluster(float pos_x, float pos_y){clusters.push_back(Cluster(pos_x, pos_y));}
    void add_element(float pos_x, float pos_y){elements.push_back(Element(pos_x, pos_y));}
    void kmeans(unsigned int iterations);
    void p_kmeans(unsigned int iterations);
    void gpu_kmeans(unsigned int iterations, unsigned int n_blocks, unsigned int n_threads_per_block);
    void gpu_print_device_info();
};

}

#endif // KMEANS_KMEANS_H
