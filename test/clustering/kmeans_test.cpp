#include <catch.hpp>
#include "kmeans_test.h"
#include <src/clustering/kmeans.h>

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

TEST_CASE_METHOD(KmeansCircuit,"Kmeans: Kmeans circuit test.", "[kmeans]"){
    KmeansCircuit::read_file("./input_files/superblue18.dat");
    generate_clusters();
    kmeans.kmeans(10);
}
