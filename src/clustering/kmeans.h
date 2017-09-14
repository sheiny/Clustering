#ifndef KMEANS_KMEANS_H
#define KMEANS_KMEANS_H

#include <vector>
#include <chrono>
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

using point = boost::geometry::model::d2::point_xy<double>;
using rtree_node = std::pair<point, std::size_t>;
using rtree = boost::geometry::index::rtree<rtree_node, boost::geometry::index::rstar<16>>;

namespace clustering{

class Kmeans{
private:
    std::vector<point> e_positions;
    std::vector<point> c_positions;
    std::vector<std::vector<point>> c_elements;
    std::size_t max_cluster_size;
    rtree clusters_rtree;

    void clear_all_clusters(){c_elements.clear(); c_elements.resize(c_positions.size());}
    void resolve_overflows();
    void clear_empty_clusters();
    void update_clusters_centers();
    void update_rtree();
    void assign_elements_2_clusters();
public:
    Kmeans(){}
    void reserve_clusters(std::size_t size){c_positions.reserve(size);}
    void reserve_elements(std::size_t size){e_positions.reserve(size);}

    void insert_cluster(double x, double y){c_positions.push_back(point(x, y));}
    void insert_element(double x, double y){e_positions.push_back(point(x, y));}

    const std::size_t size_elements() const{return e_positions.size();}
    const std::size_t size_clusters() const{return c_positions.size();}

    const std::vector<point>& cluster_elements(std::size_t c_index) const{return c_elements.at(c_index);}

    const point& element_position(std::size_t i) const{return e_positions.at(i);}
    const point& cluster_position(std::size_t i) const{return c_positions.at(i);}

    void set_max_cluster_size(std::size_t max_size){max_cluster_size = max_size;}

    void kmeans(unsigned int iterations);
};
}

#endif // KMEANS_KMEANS_H
