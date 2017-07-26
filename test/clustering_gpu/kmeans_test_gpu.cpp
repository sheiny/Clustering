#include <catch.hpp>
#include "../clustering/kmeans_fixture.h"

TEST_CASE("Kmeans: Kmeans GPU test on ICCAD 2015 circuits.", "[kmeans][gpu]"){
    for(std::string circuit_name : {"superblue18", "superblue4", "superblue16", "superblue5", "superblue1", "superblue3", "superblue10", "superblue7"}){
        KmeansCircuit circuit;
        circuit.read_file("./input_files/"+circuit_name+".dat");
        circuit.kmeans.set_max_cluster_size(50);
        const unsigned int number_of_elements(circuit.kmeans.k_elements().size());
        circuit.generate_clusters(number_of_elements/50);


        //Test overflow.
        circuit.kmeans.gpu_kmeans(1, 10, 1024);
        bool overflow = false;
        for(auto & cluster : circuit.kmeans.k_clusters())
            if(cluster.cluster_elements().size() > 50)
                overflow = true;
        REQUIRE(overflow == true);
        overflow = false;
        circuit.kmeans.gpu_kmeans(5, 10, 1024);
        for(auto & cluster : circuit.kmeans.k_clusters())
            if(cluster.cluster_elements().size() > 50)
                overflow = true;
        REQUIRE(overflow == false);
        //Test if each element was assigned to a cluster.
        unsigned int actual_number_of_elements = 0;
        for(auto cluster : circuit.kmeans.k_clusters())
            actual_number_of_elements += cluster.cluster_elements().size();
        REQUIRE(actual_number_of_elements == number_of_elements);
        //Test if each element is assigned to only one cluster.
        std::vector<unsigned int> occurrences;
        occurrences.resize(number_of_elements, 0);
        for(auto cluster : circuit.kmeans.k_clusters())
            for(auto element : cluster.cluster_elements())
                occurrences.at(element) += 1;
        bool one_to_one = true;
        for(auto mapping : occurrences)
            if(mapping != 1)
                one_to_one = false;
        REQUIRE(one_to_one == true);
        //Test if there is no one empty cluster.
        bool empty_clusters = false;
        for(auto cluster : circuit.kmeans.k_clusters())
            if(cluster.cluster_elements().empty())
                empty_clusters = true;
        REQUIRE(empty_clusters == false);
    }
}
