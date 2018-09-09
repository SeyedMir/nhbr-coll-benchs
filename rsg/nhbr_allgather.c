#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include "mpi.h"

struct time_stats {
    double total;
    double average;
};

#ifdef ERROR_CHECK
    #define Type_MPI MPI_INT
    typedef int Datatype;
#else
    #define Type_MPI MPI_CHAR
    typedef char Datatype;
#endif

#define LARGE_MSG_THR 1024
#define LARGE_MSG_ITR 300

static int make_nhbrhood(int my_rank, double p, int size, MPI_Comm comm,
                         int *indegree_ptr,
                         int **sources_ptr,
                         int **sourcesweights_ptr,
                         int *outdegree_ptr,
                         int **destinations_ptr,
                         int **destweights_ptr)
{
    int i, j, indgr, outdgr, inidx, outidx;
    indgr = outdgr = inidx = outidx = 0;
    int *srcs, *srcwghts, *dests, *destwghts, *my_row, *my_col;

    my_row = (int*) calloc(size, sizeof(int));
    my_col = (int*) calloc(size, sizeof(int));

    int *contig_vtopo_mat = NULL;
    int *verify_contig_vtopo_mat = NULL;
    int **vtopo_mat = NULL;

    //Rank 0 builds the random graph and scatters it to others
    if(my_rank == 0)
    {
        contig_vtopo_mat = (int*) calloc(size * size, sizeof(int));
        verify_contig_vtopo_mat = (int*) calloc(size * size, sizeof(int));
        vtopo_mat = (int**) malloc(size * sizeof(int*));
        for(i = 0; i < size; i++)
        {
            //vtopo_mat[i] = (int*) calloc(size, sizeof(int));
            vtopo_mat[i] = &contig_vtopo_mat[i*size];
        }

        //Making the random graph
        srand(time(NULL));
        double x;
        for(i = 0; i < size; i++)
        {
            for(j = 0; j < size; j++)
            {
                if(i == j) continue;
                x = (double)rand() / (double)RAND_MAX;
                if(x < p)
                    vtopo_mat[i][j] = 1;
            }
        }
    }

    //Scattering rows of the matrix
    MPI_Scatter(contig_vtopo_mat, size, MPI_INT, my_row, size, MPI_INT, 0, comm);

    //Scattering columns of the matrix
    MPI_Datatype mat_col_t, mat_col_resized_t;
    MPI_Type_vector(size, 1, size, MPI_INT, &mat_col_t);
    MPI_Type_commit(&mat_col_t);
    MPI_Type_create_resized(mat_col_t, 0, 1*sizeof(int), &mat_col_resized_t);
    MPI_Type_commit(&mat_col_resized_t);
    MPI_Scatter(contig_vtopo_mat, 1, mat_col_resized_t, my_col, size, MPI_INT, 0, comm);

#ifdef ERROR_CHECK
    //Verify the correctness of matrix distribution
    MPI_Gather(my_row, size, MPI_INT,
               verify_contig_vtopo_mat, size, MPI_INT, 0, comm);
    if(my_rank == 0)
    {
        for(i = 0; i < size*size; i++)
        {
            if(contig_vtopo_mat[i] != verify_contig_vtopo_mat[i])
            {
                fprintf(stderr, "ERROR: contig matrix is not same "
                                "as the aggregation of all my_rows!\n");
                return 1;
            }
        }
        memset(verify_contig_vtopo_mat, 0, size * size * sizeof(int));
    }

    MPI_Gather(my_col, size, MPI_INT,
               verify_contig_vtopo_mat, 1, mat_col_resized_t, 0, comm);
    if(my_rank == 0)
    {
        for(i = 0; i < size*size; i++)
        {
            if(contig_vtopo_mat[i] != verify_contig_vtopo_mat[i])
            {
                fprintf(stderr, "ERROR: contig matrix is not "\
                                "same as the aggregation of all my_cols!\n");
                return 1;
            }
        }
    }
#endif

    //free some memory
    if(my_rank == 0)
    {
        free(vtopo_mat);
        free(contig_vtopo_mat);
        free(verify_contig_vtopo_mat);
    }
    MPI_Type_free(&mat_col_resized_t);
    MPI_Type_free(&mat_col_t);

    //Finding indegree and outdegree
    for(i = 0; i < size; i++)
    {
        if(my_row[i] != 0)
            outdgr++;
        if(my_col[i] != 0)
            indgr++;
    }

    srcs = (int*) malloc(indgr * sizeof(int));
    srcwghts = (int*) malloc(indgr * sizeof(int));
    dests = (int*) malloc(outdgr * sizeof(int));
    destwghts = (int*) malloc(outdgr * sizeof(int));

    for(i = 0; i < indgr; i++)
        srcwghts[i] = 1;
    for(i = 0; i < outdgr; i++)
        destwghts[i] = 1;

    for(i = 0; i < size; i++)
    {
        if(my_row[i] != 0)
        {
            dests[outidx] = i;
            outidx++;
        }
        if(my_col[i] != 0)
        {
            srcs[inidx] = i;
            inidx++;
        }
    }

    //Returning all values
    *indegree_ptr = indgr;
    *sources_ptr = srcs;
    *sourcesweights_ptr = srcwghts;
    *outdegree_ptr = outdgr;
    *destinations_ptr = dests;
    *destweights_ptr = destwghts;

    free(my_row);
    free(my_col);

    return 0;
}

static struct time_stats iterate_coll(int itr,
                                      Datatype *sendbuf, int sendcount,
                                      Datatype *recvbuf, int recvcount,
                                      MPI_Comm nhbr_comm)
{
    int skip = itr / 10;
    double start_time = 0;
    int i;
    for(i = 0; i < itr; i++)
    {
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
    //initialize sendbuf
    for(i = 0; i < max_msg_size; i++)
	    sendbuf[i] = my_rank;
#endif

    for(i = 0; i < num_msg_sizes; i++)
    {
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
        for(j = 0; j < indgr; j++)
        {
            if(recvbuf[j] != srcs[j])
            {
                fprintf(stderr,
                        "Rank %d: ERROR: mismatch between recv buffer "
                        "and srcs at index %d\n", my_rank, j);
                break;
            }
        }
#endif

        if(my_rank == 0)
        {
            printf("Total communication time = %lf  (s)\n", time_stats.total);
            printf("Single iteration average time = %lf  (s)\n\n", time_stats.average);
        }
    }

    free(sendbuf);
    free(recvbuf);
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
    if(my_rank == 0 || my_rank == 1)
    {
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

    if(argc != 3)
    {
        if(my_rank == 0)
            fprintf(stderr, "Usage: exec_file <sparsity factor (0 < p < 1)> "\
                            "<number of iterations>\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    double p = atof(argv[1]);
	int itr = atoi(argv[2]);

#ifdef ERROR_CHECK
    char *out_dir = "rank_output_files";
    char *cwd = getcwd(NULL, 0);
    if(cwd == NULL)
    {
        fprintf(stderr, "Rank %d: Failed to get the current working directory\n",
                my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    char out_file[256];
    snprintf(out_file, sizeof(out_file), "%s/%d", out_dir, my_rank);
    FILE *fp = fopen(out_file, "w");
    if(fp == NULL)
    {
        fprintf(stderr, "Failed to open file %s/%s\n", cwd, out_file);
        free(cwd);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
#endif

    //Defining the neighborhood
    int indgr, outdgr;
    int *srcs, *srcwghts, *dests, *destwghts;
    int err = make_nhbrhood(my_rank, p, comm_size, MPI_COMM_WORLD,
                           &indgr, &srcs, &srcwghts,
                           &outdgr, &dests, &destwghts);
    if(err)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

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

    //Create a communicator with the topology information attached
    MPI_Comm nhbr_comm;
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                   indgr, srcs, srcwghts,
                                   outdgr, dests, destwghts,
                                   MPI_INFO_NULL, 0, &nhbr_comm);

    run_loop(itr, indgr, outdgr, my_rank, nhbr_comm);

#ifdef ERROR_CHECK
    fclose(fp);
    free(cwd);
#endif

    MPI_Comm_free(&nhbr_comm);
    MPI_Finalize();
    return 0;
}
