#include <catch.hpp>
#include "kmeans_test.h"
#include <src/clustering/kmeans.h>

class KmeansFixture {
public:
    KmeansFixture(){
        kmeans.reserve_clusters(4);
        kmeans.reserve_elements(21);

        kmeans.add_cluster(3, 3);
        kmeans.add_cluster(3, 8);
        kmeans.add_cluster(9, 2);
        kmeans.add_cluster(9, 8);
        kmeans.add_cluster(12, 5);

        kmeans.add_element(1, 1);//1
        kmeans.add_element(2, 3);
        kmeans.add_element(2, 7);
        kmeans.add_element(2, 9);
        kmeans.add_element(3, 6);//5
        kmeans.add_element(4, 4);
        kmeans.add_element(4, 8.5);
        kmeans.add_element(4.5, 2);
        kmeans.add_element(4.5, 9.5);
        kmeans.add_element(5, 6);//10
        kmeans.add_element(5.5, 7.5);
        kmeans.add_element(7.5, 1.5);
        kmeans.add_element(7, 8);
        kmeans.add_element(8.5, 0.5);
        kmeans.add_element(8, 7);//15
        kmeans.add_element(8, 10);
        kmeans.add_element(9, 4);
        kmeans.add_element(10, 5);
        kmeans.add_element(10, 9);
        kmeans.add_element(11, 4);//20
        kmeans.add_element(11, 11);
    }
    clustering::Kmeans kmeans;
};

TEST_CASE_METHOD(KmeansFixture,"Kmeans: Kmeans test.", "[kmeans]")
{
    REQUIRE(kmeans.cluster(0).x() == 3.0);
    REQUIRE(kmeans.cluster(0).y() == 3.0);
    REQUIRE(kmeans.cluster(1).x() == 3.0);
    REQUIRE(kmeans.cluster(1).y() == 8.0);
    REQUIRE(kmeans.cluster(2).x() == 9.0);
    REQUIRE(kmeans.cluster(2).y() == 2.0);
    REQUIRE(kmeans.cluster(3).x() == 9.0);
    REQUIRE(kmeans.cluster(3).y() == 8.0);
    REQUIRE(kmeans.cluster(4).x() == 12.0);
    REQUIRE(kmeans.cluster(4).y() == 5.0);

    kmeans.kmeans(1);

    REQUIRE(kmeans.cluster(0).x() == 2.875);
    REQUIRE(kmeans.cluster(0).y() == 2.5);
    REQUIRE(kmeans.cluster(1).x() == Approx(3.71429));
    REQUIRE(kmeans.cluster(1).y() == Approx(7.64286));
    REQUIRE(kmeans.cluster(2).x() == Approx(8.33333));
    REQUIRE(kmeans.cluster(2).y() == 2);
    REQUIRE(kmeans.cluster(3).x() == Approx(8.8));
    REQUIRE(kmeans.cluster(3).y() == 9);
    REQUIRE(kmeans.cluster(4).x() == 10.5);
    REQUIRE(kmeans.cluster(4).y() == 4.5);

    kmeans.kmeans(1);

    REQUIRE(kmeans.cluster(0).x() == 2.875);
    REQUIRE(kmeans.cluster(0).y() == 2.5);
    REQUIRE(kmeans.cluster(1).x() == Approx(3.71429));
    REQUIRE(kmeans.cluster(1).y() == Approx(7.64286));
    REQUIRE(kmeans.cluster(2).x() == 8);
    REQUIRE(kmeans.cluster(2).y() == 1);
    REQUIRE(kmeans.cluster(3).x() == Approx(8.8));
    REQUIRE(kmeans.cluster(3).y() == 9);
    REQUIRE(kmeans.cluster(4).x() == 10);
    REQUIRE(kmeans.cluster(4).y() == Approx(4.33333));
}
