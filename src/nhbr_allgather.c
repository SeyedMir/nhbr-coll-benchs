#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <getopt.h>
#include "mpi.h"
#include "nhbr_topo.h"
#include "bench.h"

struct bench_config {
    int itr;
    struct nhbrhood_config nhbrhood_config;
};

struct time_stats {
    double total;
    double average;
};

static struct time_stats iterate_coll(int itr,
                                      Datatype *sendbuf, int sendcount,
                                      Datatype *recvbuf, int recvcount,
                                      MPI_Comm nhbr_comm)
{
    int skip = itr / 10;
    double start_time = 0;
    int i;
    for(i = 0; i < itr; i++) {
        if(i == skip)
            start_time = MPI_Wtime();

        MPI_Request req;
        MPI_Ineighbor_allgather(sendbuf, sendcount, Type_MPI,
                                recvbuf, recvcount, Type_MPI,
                                nhbr_comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    struct time_stats time_stats = {0};
    time_stats.total = end_time - start_time;
    time_stats.average = time_stats.total / (itr - skip);

    return time_stats;
}

static void run_loop(int itr, int indgr, int outdgr, int my_rank, MPI_Comm nhbr_comm)
{
#ifdef ERROR_CHECK
	int num_msg_sizes = 1;
    int msg_sizes[1] = {1};
#else
	int num_msg_sizes = 8;
    int msg_sizes[8] = {1, 4, 16, 64, 256, 1024, 4096, 16384};
#endif

    int max_msg_size = msg_sizes[num_msg_sizes - 1];
    Datatype *sendbuf = (Datatype*) calloc(max_msg_size, sizeof(Datatype));
    Datatype *recvbuf = (Datatype*) calloc(indgr * max_msg_size, sizeof(Datatype));

    int i;
#ifdef ERROR_CHECK
    /* initialize sendbuf */
    for(i = 0; i < max_msg_size; i++)
	    sendbuf[i] = my_rank;
#endif

    for(i = 0; i < num_msg_sizes; i++) {
        int msg_size = msg_sizes[i];
        if(my_rank == 0)
            printf("------ Starting the experimet with %d Byte(s) ------\n", msg_size);

        memset(recvbuf, 0, indgr * max_msg_size * sizeof(Datatype));
        struct time_stats time_stats = {0};
        if(msg_size > LARGE_MSG_THR && itr > LARGE_MSG_ITR)
            time_stats = iterate_coll(LARGE_MSG_ITR,
                                      sendbuf, msg_size, recvbuf, msg_size, nhbr_comm);
        else
            time_stats = iterate_coll(itr,
                                      sendbuf, msg_size, recvbuf, msg_size, nhbr_comm);

#ifdef ERROR_CHECK
        int j;
        for(j = 0; j < indgr; j++) {
            if(recvbuf[j] != srcs[j]) {
                fprintf(stderr,
                        "Rank %d: ERROR: mismatch between recv buffer "
                        "and srcs at index %d\n", my_rank, j);
                break;
            }
        }
#endif

        if(my_rank == 0) {
            printf("Total communication time = %lf  (s)\n", time_stats.total);
            printf("Single iteration average time = %lf  (s)\n\n", time_stats.average);
        }
    }

    free(sendbuf);
    free(recvbuf);
}

static int int_from_string(const char *str, int *number)
{
    int ret = 0;
    errno = 0;
    char *endptr;
    *number = (int)strtol(str, &endptr, 10);
    if(errno != 0 || endptr == str)
        ret = -1;

    return ret;
}

static int double_from_string(const char *str, double *number)
{
    errno = 0;
    char *endptr;
    *number = strtod(str, &endptr);
    if(errno != 0 || endptr == str) {
        free(endptr);
        return -1;
    }

    return 0;
}

static int topo_from_string(const char *str, enum Topology *config_topo)
{
    if(strcmp(str, "rsg") == 0)
        *config_topo = RSG;
    else if(strcmp(str, "moore") == 0)
        *config_topo = MOORE;
    else
        return -1;

    return 0;
}

static int parse_arguments(int my_rank, int argc, char **argv, struct bench_config *config)
{
    static const char *usage =
        "mpiexec -n <num_procs> nhbr_allgather [options]\n"
        "Options:\n"
        "    -n, --itr <iterations>   Number of iterations to call neighborhood collective.\n"
        "                             Default is 1000.\n"
        "\n"
        "    -t , --topo <rsg|moore>  Choose the neghborhood topology. Default is moore.\n"
        "\n"
        "    -d, --dim <dimension>    Dimension of the moore topology. Default is 2.\n"
        "\n"
        "    -r, --radius <radius>    Radius value of the moore topology. Default is 2.\n"
        "\n"
        "    -p, --prob <sparsity>    Sparsity factor for the random sparse graph topology.\n"
        "                             Must be used  with '-t rsg' option.\n"
        "\n"
        "    -h, --help               Show this usage message.\n"
        "\n";

    struct option opts[] = {
        {"itr", required_argument, NULL, 'n'},
        {"topo", required_argument, NULL, 't'},
        {"dim", required_argument, NULL, 'd'},
        {"radius", required_argument, NULL, 'r'},
        {"prob", required_argument, NULL, 'p'},
        {"help", no_argument, NULL, 'h'},
    };

    /* Set config defaults */
    config->itr = 1000;
    config->nhbrhood_config.topo = MOORE;
    config->nhbrhood_config.topo_params.moore_params.d = 2;
    config->nhbrhood_config.topo_params.moore_params.r = 2;

    int c;
    while((c = getopt_long(argc, argv, "n:t:d:r:p:h", opts, NULL)) != -1) {
        union topo_params *topo_params = &config->nhbrhood_config.topo_params;
        switch (c) {
            case 'n':
                if(int_from_string(optarg, &config->itr) != 0)
                    return -1;
                break;
            case 't':
                if(topo_from_string(optarg, &config->nhbrhood_config.topo) != 0)
                    return -1;
                break;
            case 'd':
                if(int_from_string(optarg, &topo_params->moore_params.d) != 0)
                    return -1;
                break;
            case 'r':
                if(int_from_string(optarg, &topo_params->moore_params.r) != 0)
                    return -1;
                break;
            case 'p':
                if(double_from_string(optarg, &topo_params->rsg_params.p) != 0)
                    return -1;
                break;
            case 'h':
                if(my_rank == 0) printf("%s", usage);
                MPI_Finalize();
                exit(EXIT_SUCCESS);
            default:
                if(my_rank == 0) fprintf(stderr, "Invalid option\n");
                return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* For gdb process attachment */
    /*
    i = 0;
    if(my_rank == 0 || my_rank == 1) {
        char hostname[256];
        printf("sizeof hostname = %u\n", sizeof(hostname));
        gethostname(hostname, sizeof(hostname));
        printf("PID %d (rank %d) on %s ready for attach\n",
               getpid(), my_rank, hostname);
        fflush(stdout);
        while(i == 0)
            sleep(5);
    }
    */

    struct bench_config config = {0};
    if(parse_arguments(my_rank, argc, argv, &config) != 0) {
        if(my_rank == 0)
            fprintf(stderr, "Bad options or arguments\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#ifdef ERROR_CHECK
    char *out_dir = "rank_output_files";
    char *cwd = getcwd(NULL, 0);
    if(cwd == NULL) {
        fprintf(stderr, "Rank %d: Failed to get the current working directory\n",
                my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    char out_file[256];
    snprintf(out_file, sizeof(out_file), "%s/%d", out_dir, my_rank);
    FILE *fp = fopen(out_file, "w");
    if(fp == NULL) {
        fprintf(stderr, "Failed to open file %s/%s\n", cwd, out_file);
        free(cwd);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif

    /* Defining the neighborhood */
    int indgr, outdgr;
    int *srcs, *srcwghts, *dests, *destwghts;
    int err = make_nhbrhood(MPI_COMM_WORLD, config.nhbrhood_config,
                            &indgr, &srcs, &srcwghts,
                            &outdgr, &dests, &destwghts);
    if(err)
        MPI_Abort(MPI_COMM_WORLD, 2);

	if(my_rank == 0)
        printf("Rank %d: indegree = %d, outdegree = %d\n",
                my_rank, indgr, outdgr);

#ifdef ERROR_CHECK
    fprintf(fp, "indegree = %d, outdegree = %d\n", indgr, outdgr);

    fprintf(fp, "source ranks:\n");
    for(j = 0; j < indgr; j++)
        fprintf(fp, "%d ", srcs[j]);
    fprintf(fp, "\n");

    fprintf(fp, "destination ranks:\n");
    for(j = 0; j < outdgr; j++)
        fprintf(fp, "%d ", dests[j]);
    fprintf(fp, "\n");
#endif

    /* Create a communicator with the topology information attached */
    MPI_Comm nhbr_comm;
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                   indgr, srcs, srcwghts,
                                   outdgr, dests, destwghts,
                                   MPI_INFO_NULL, 0, &nhbr_comm);

    run_loop(config.itr, indgr, outdgr, my_rank, nhbr_comm);

#ifdef ERROR_CHECK
    fclose(fp);
    free(cwd);
#endif

    MPI_Comm_free(&nhbr_comm);
    MPI_Finalize();
    return 0;
}
