#ifndef KMEANS_FIXTURE_H
#define KMEANS_FIXTURE_H

#include <src/clustering/kmeans.h>
#include <fstream>
#include <stdlib.h>

class KmeansFixture {
public:
    KmeansFixture(){
        kmeans.reserve_clusters(4);
        kmeans.reserve_elements(21);

        kmeans.insert_cluster(3, 3);
        kmeans.insert_cluster(3, 8);
        kmeans.insert_cluster(9, 2);
        kmeans.insert_cluster(9, 8);
        kmeans.insert_cluster(12, 5);

        kmeans.insert_element(1, 1);
        kmeans.insert_element(2, 3);
        kmeans.insert_element(2, 7);
        kmeans.insert_element(2, 9);
        kmeans.insert_element(3, 6);
        kmeans.insert_element(4, 4);
        kmeans.insert_element(4, 8.5);
        kmeans.insert_element(4.5, 2);
        kmeans.insert_element(4.5, 9.5);
        kmeans.insert_element(5, 6);
        kmeans.insert_element(5.5, 7.5);
        kmeans.insert_element(7.5, 1.5);
        kmeans.insert_element(7, 8);
        kmeans.insert_element(8.5, 0.5);
        kmeans.insert_element(8, 7);
        kmeans.insert_element(8, 10);
        kmeans.insert_element(9, 4);
        kmeans.insert_element(10, 5);
        kmeans.insert_element(10, 9);
        kmeans.insert_element(11, 4);
        kmeans.insert_element(11, 11);
    }
    clustering::Kmeans kmeans;
};

class KmeansCircuit {
public:
    KmeansCircuit(){
    }

    void read_file(std::string file_path){
        std::ifstream input_file(file_path);
        unsigned int n_elements = 0;
        std::string word;
        while(input_file>>word){
            input_file>>word;
            ++n_elements;
        }
        kmeans.reserve_elements(n_elements);
        input_file.close();
        input_file.open(file_path);
        while(input_file>>word){
            double x = std::stod(word);
            input_file>>word;
            double y = std::stod(word);
            kmeans.insert_element(x, y);
        }
    }

    void generate_clusters(unsigned int n_clusters){
        std::pair<double, double> max_coordinate(0, 0);
        for(std::size_t e_index = 0; e_index < kmeans.size_elements(); e_index++){
            max_coordinate.first = std::max(max_coordinate.first,  kmeans.element_position(e_index).x());
            max_coordinate.second = std::max(max_coordinate.second,  kmeans.element_position(e_index).y());
        }
        srand(42);
        kmeans.reserve_clusters(n_clusters);
        for(unsigned int i = 0; i < n_clusters; ++i)
            kmeans.insert_cluster(rand()%(int)max_coordinate.first, rand()%(int)max_coordinate.second);
    }

    clustering::Kmeans kmeans;
};

#endif // KMEANS_FIXTURE_H
