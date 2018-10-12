#ifndef NHBR_TOPO_H
#define NHBR_TOPO_H

enum Topology {
    RSG,    /* Random Sparse Graph */
    MOORE,  /* Moore neighborhodd */
};

struct rsg_params {
    float p;
};

struct moore_params {
    int d;
    int r;
};

union topo_params {
    struct rsg_params rsg_params;
    struct moore_params moore_params;
};

struct nhbrhood_config {
    enum Topology topo;
    union topo_params topo_params;
};

int make_nhbrhood(MPI_Comm comm, struct nhbrhood_config config,
                  int *indegree_ptr, int **sources_ptr, int **sourcesweights_ptr,
                  int *outdegree_ptr, int **destinations_ptr, int **destweights_ptr);
#endif
