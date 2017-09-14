#include <catch.hpp>
#include "kmeans_fixture.h"

TEST_CASE_METHOD(KmeansFixture,"Kmeans: Kmeans test.", "[kmeans]")
{
    kmeans.set_max_cluster_size(42);

    REQUIRE(kmeans.cluster_position(0).x() == 3.0);
    REQUIRE(kmeans.cluster_position(0).y() == 3.0);
    REQUIRE(kmeans.cluster_position(1).x() == 3.0);
    REQUIRE(kmeans.cluster_position(1).y() == 8.0);
    REQUIRE(kmeans.cluster_position(2).x() == 9.0);
    REQUIRE(kmeans.cluster_position(2).y() == 2.0);
    REQUIRE(kmeans.cluster_position(3).x() == 9.0);
    REQUIRE(kmeans.cluster_position(3).y() == 8.0);
    REQUIRE(kmeans.cluster_position(4).x() == 12.0);
    REQUIRE(kmeans.cluster_position(4).y() == 5.0);

    kmeans.kmeans(1);

    REQUIRE(kmeans.cluster_position(0).x() == 2.875);
    REQUIRE(kmeans.cluster_position(0).y() == 2.5);
    REQUIRE(kmeans.cluster_position(1).x() == Approx(3.71429));
    REQUIRE(kmeans.cluster_position(1).y() == Approx(7.64286));
    REQUIRE(kmeans.cluster_position(2).x() == Approx(8.33333));
    REQUIRE(kmeans.cluster_position(2).y() == 2);
    REQUIRE(kmeans.cluster_position(3).x() == Approx(8.8));
    REQUIRE(kmeans.cluster_position(3).y() == 9);
    REQUIRE(kmeans.cluster_position(4).x() == 10.5);
    REQUIRE(kmeans.cluster_position(4).y() == 4.5);

    kmeans.kmeans(1);

    REQUIRE(kmeans.cluster_position(0).x() == 2.875);
    REQUIRE(kmeans.cluster_position(0).y() == 2.5);
    REQUIRE(kmeans.cluster_position(1).x() == Approx(3.71429));
    REQUIRE(kmeans.cluster_position(1).y() == Approx(7.64286));
    REQUIRE(kmeans.cluster_position(2).x() == 8);
    REQUIRE(kmeans.cluster_position(2).y() == 1);
    REQUIRE(kmeans.cluster_position(3).x() == Approx(8.8));
    REQUIRE(kmeans.cluster_position(3).y() == 9);
    REQUIRE(kmeans.cluster_position(4).x() == 10);
    REQUIRE(kmeans.cluster_position(4).y() == Approx(4.33333));
}

TEST_CASE("Kmeans: Resolve overflow.", "[kmeans]"){
    KmeansCircuit circuit;

    circuit.kmeans.insert_cluster(0, 0);

    circuit.kmeans.insert_element(3, 6);
    circuit.kmeans.insert_element(5, 1);
    circuit.kmeans.insert_element(7, 1);
    circuit.kmeans.insert_element(2, 3);
    circuit.kmeans.insert_element(1, 8);

    circuit.kmeans.set_max_cluster_size(2);

    circuit.kmeans.kmeans(1);

    REQUIRE(circuit.kmeans.size_clusters() == 3);
    std::vector<std::size_t> expected_sizes = {1, 2, 2};
    std::vector<std::size_t> actual_clusters_sizes;
    for(std::size_t c_index = 0; c_index < circuit.kmeans.size_clusters(); c_index++)
        actual_clusters_sizes.push_back(circuit.kmeans.cluster_elements(c_index).size());
    REQUIRE(std::is_permutation(actual_clusters_sizes.begin(), actual_clusters_sizes.end(), expected_sizes.begin()));
}

TEST_CASE("Kmeans: Kmeans test on ICCAD 2015 circuits.", "[kmeans]"){
    for(std::string circuit_name : {"superblue18", "superblue4", "superblue16", "superblue5", "superblue1", "superblue3", "superblue10", "superblue7"}){
        KmeansCircuit circuit;
        circuit.read_file("./input_files/"+circuit_name+".dat");
        std::size_t max_cluster_size = 50;
        circuit.kmeans.set_max_cluster_size(max_cluster_size);
        const unsigned int number_of_elements(circuit.kmeans.size_elements());
        circuit.generate_clusters(number_of_elements/50);

        circuit.kmeans.kmeans(1);

        //Test overflow.
        bool overflow = false;
        for(std::size_t c_index = 0; c_index < circuit.kmeans.size_clusters(); c_index++)
            if(circuit.kmeans.cluster_elements(c_index).size() > max_cluster_size)
                overflow = true;
        REQUIRE(overflow == false);
//        //Test if each element was assigned to a cluster.
//        unsigned int actual_number_of_elements = 0;
//        for(auto cluster : circuit.kmeans.k_clusters())
//            actual_number_of_elements += cluster.cluster_elements().size();
//        REQUIRE(actual_number_of_elements == number_of_elements);
//        //Test if each element is assigned to only one cluster.
//            std::vector<unsigned int> occurrences;
//            occurrences.resize(number_of_elements, 0);
//            for(auto cluster : circuit.kmeans.k_clusters())
//                for(auto element : cluster.cluster_elements())
//                    occurrences.at(element) += 1;
//            bool one_to_one = true;
//            for(auto mapping : occurrences)
//                if(mapping != 1)
//                    one_to_one = false;
//            REQUIRE(one_to_one == true);
        //Test if there is no one empty cluster.
            bool empty_clusters = false;
            for(std::size_t c_index = 0; c_index < circuit.kmeans.size_clusters(); c_index++)
                if(circuit.kmeans.cluster_elements(c_index).empty())
                    empty_clusters = true;
            REQUIRE(empty_clusters == false);
    }
}

TEST_CASE("Kmeans: Kmeans circuit test sequential.", "[sequential]"){
    for(unsigned int i = 0; i < 10; ++i){
        for(std::string circuit_name : {"superblue18", "superblue4", "superblue16", "superblue5", "superblue1", "superblue3", "superblue10", "superblue7"}){
            KmeansCircuit sequential;
            sequential.kmeans.set_max_cluster_size(50);
            sequential.read_file("./input_files/"+circuit_name+".dat");
            std::cout<<circuit_name<<" ";
            sequential.generate_clusters(sequential.kmeans.size_elements()/50);
            sequential.kmeans.kmeans(50);
            std::cout<<" k "<<sequential.kmeans.size_clusters()<<std::endl;
        }
    }
}
