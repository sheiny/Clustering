#include <catch.hpp>
#include "kmeans_test_gpu.h"
#include <src/clustering/kmeans.h>

TEST_CASE("GPU: test"){
    clustering::Kmeans k;
    k.gpu();
}
