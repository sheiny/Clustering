#include "kmeans_fixture.h"

namespace fixture{
bool element_comparison(const clustering::Element & e1, const clustering::Element & e2){return e1.cluster() == e2.cluster();}
bool cluster_comparison(const clustering::Cluster & c1, const clustering::Cluster & c2){return c1.x() == c2.x() && c1.y() == c2.y();}
bool cluster_assignment_comparsion(const clustering::Element & e1, const clustering::Element & e2){return e1.x() == e2.x() && e1.y() == e2.y();}
}
