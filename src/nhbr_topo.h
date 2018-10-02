#ifndef NHBR_TOPO_H
#define NHBR_TOPO_H

enum Topology {
    RSG,    /* Random Sparse Graph */
    MOORE,  /* Moore neighborhodd */
};

struct nhbrhood_config {
    enum Topology topo;
    int d;
    int r;
    float p;
};

int make_nhbrhood(MPI_Comm comm, struct nhbrhood_config config,
                  int *indegree_ptr, int **sources_ptr, int **sourcesweights_ptr,
                  int *outdegree_ptr, int **destinations_ptr, int **destweights_ptr);
#endif
