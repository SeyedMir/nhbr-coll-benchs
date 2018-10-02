#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "nhbr_topo.h"

static int make_rsg_nhbrhood(MPI_Comm comm, float p,
        int *indegree_ptr, int **sources_ptr, int **sourcesweights_ptr,
        int *outdegree_ptr, int **destinations_ptr, int **destweights_ptr)
{
    int my_rank, size;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &size);

    int i, j, indgr, outdgr, inidx, outidx;
    indgr = outdgr = inidx = outidx = 0;

    int *my_row, *my_col;
    my_row = (int*) calloc(size, sizeof(int));
    my_col = (int*) calloc(size, sizeof(int));

    int *contig_vtopo_mat = NULL;
    int *verify_contig_vtopo_mat = NULL;
    int **vtopo_mat = NULL;

    /* Rank 0 builds the random graph and scatters it to others */
    if(my_rank == 0)
    {
        contig_vtopo_mat = (int*) calloc(size * size, sizeof(int));
        verify_contig_vtopo_mat = (int*) calloc(size * size, sizeof(int));
        vtopo_mat = (int**) malloc(size * sizeof(int*));
        for(i = 0; i < size; i++)
            vtopo_mat[i] = &contig_vtopo_mat[i*size];

        /* Making the random graph */
        srand(time(NULL));
        float x;
        for(i = 0; i < size; i++)
        {
            for(j = 0; j < size; j++)
            {
                if(i == j) continue;
                x = (float)rand() / RAND_MAX;
                if(x < p)
                    vtopo_mat[i][j] = 1;
            }
        }
    }

    /* Scattering rows of the matrix */
    MPI_Scatter(contig_vtopo_mat, size, MPI_INT, my_row, size, MPI_INT, 0, comm);

    /* Scattering columns of the matrix */
    MPI_Datatype mat_col_t, mat_col_resized_t;
    MPI_Type_vector(size, 1, size, MPI_INT, &mat_col_t);
    MPI_Type_commit(&mat_col_t);
    MPI_Type_create_resized(mat_col_t, 0, 1*sizeof(int), &mat_col_resized_t);
    MPI_Type_commit(&mat_col_resized_t);
    MPI_Scatter(contig_vtopo_mat, 1, mat_col_resized_t, my_col, size, MPI_INT, 0, comm);

#ifdef ERROR_CHECK
    /* Verify the correctness of matrix distribution */
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

    /* free some memory */
    if(my_rank == 0)
    {
        free(vtopo_mat);
        free(contig_vtopo_mat);
        free(verify_contig_vtopo_mat);
    }
    MPI_Type_free(&mat_col_resized_t);
    MPI_Type_free(&mat_col_t);

    /* Finding indegree and outdegree */
    for(i = 0; i < size; i++)
    {
        if(my_row[i] != 0)
            outdgr++;
        if(my_col[i] != 0)
            indgr++;
    }

    int *srcs, *srcwghts, *dests, *destwghts;
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

    /* Returning all values */
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

static int array_add(const int *a, const int *b, int size, int *c)
{
    int i;
    for(i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
    return 0;
}

static int make_moore_nhbrhood(MPI_Comm comm, int d, int r,
        int *indegree_ptr, int **sources_ptr, int **sourcesweights_ptr,
        int *outdegree_ptr, int **destinations_ptr, int **destweights_ptr)
{
    int i, indgr, outdgr, inidx, outidx, comm_size, my_rank, nhbr_rank, min_dim;
    int *srcs, *srcwghts, *dests, *destwghts;
    int dims[d];
    int periods[d];
    int my_coords[d];
    int nhbr_coords[d];
    int disp_vec[d];
    MPI_Comm cart_comm;

    indgr = outdgr = inidx = outidx = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &my_rank);

    /* Set the periods, and initialize dims to 0 */
    for(i = 0; i < d; i++)
    {
        periods[i] = 1;
        dims[i] = 0;
    }
    MPI_Dims_create(comm_size, d, dims);
    MPI_Cart_create(comm, d, dims, periods, 0, &cart_comm);


    /* Print the dimension sizes */
    if(my_rank == 0)
    {
        printf("dims = ");
        for(i = 0; i < d; i++)
        {
            printf("%d ", dims[i]);
        }
        printf("\n");
    }

    /* Find max valid r based on minimum dimension size */
    min_dim = comm_size;
    for(i = 0; i < d; i++)
    {
        if(dims[i] < min_dim)
            min_dim = dims[i];
    }
    if(r > ((min_dim - 1) / 2)) /* Divided by 2 to avoid duplicate neighbors */
    {
        if(my_rank == 0)
        {
            printf("ERROR: the given neighborhood radius (r = %d) is greater than "
                   "half of the minimum dimension size %d. Aborting!\n", r, min_dim);
            fflush(stdout);
        }
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        exit(0);
    }

    /* Calculate number of neighbors */
    outdgr = indgr = pow((2*r + 1), d) - 1;

    srcs = (int*) malloc(indgr * sizeof(int));
    srcwghts = (int*) malloc(indgr * sizeof(int));
    dests = (int*) malloc(outdgr * sizeof(int));
    destwghts = (int*) malloc(outdgr * sizeof(int));

    for(i = 0; i < indgr; i++)
        srcwghts[i] = 1;
    for(i = 0; i < outdgr; i++)
        destwghts[i] = 1;

    /* Initialize the displacement vector */
    for(i = 0; i < d; i++)
        disp_vec[i] = -r;

    MPI_Cart_coords(cart_comm, my_rank, d, my_coords);
    int overflow = 0;
    while(!overflow)
    {
        /* The displacement vector will act like a counter
         * that is increased in each iteration to find the
         * next neighbor. Each digit of the counter spans
         * from -r to r.
         */
        array_add(my_coords, disp_vec, d, nhbr_coords);
        MPI_Cart_rank(cart_comm, nhbr_coords, &nhbr_rank);
        if(nhbr_rank != my_rank) /* Skip the case where nhbr_coords is equal to all 0 */
        {
            dests[outidx] = nhbr_rank;
            srcs[inidx] = nhbr_rank;
            outidx++;
            inidx++;
        }

        /* Increase displacement vector by 1 */
        for(i = d - 1; i >= -1; i--)
        {
            if(i == -1)
            {
                overflow = 1;
                break;
            }

            if(disp_vec[i] == r) /* Have carry, do not break */
            {
                disp_vec[i] = -r;
            }
            else
            {
                disp_vec[i]++;
                break;
            }
        }
    }

    /* Returning all values */
    *indegree_ptr = indgr;
    *sources_ptr = srcs;
    *sourcesweights_ptr = srcwghts;
    *outdegree_ptr = outdgr;
    *destinations_ptr = dests;
    *destweights_ptr = destwghts;

    MPI_Comm_free(&cart_comm);
    return 0;
}

int make_nhbrhood(MPI_Comm comm, struct nhbrhood_config config,
                  int *indegree_ptr, int **sources_ptr, int **sourcesweights_ptr,
                  int *outdegree_ptr, int **destinations_ptr, int **destweights_ptr)
{
    if(config.topo == RSG)
    {
        return make_rsg_nhbrhood(comm, config.p,
                                 indegree_ptr, sources_ptr, sourcesweights_ptr,
                                 outdegree_ptr, destinations_ptr, destweights_ptr);
    } else if(config.topo == MOORE)
    {
        return make_moore_nhbrhood(comm, config.d, config.r,
                                   indegree_ptr, sources_ptr, sourcesweights_ptr,
                                   outdegree_ptr, destinations_ptr, destweights_ptr);
    }
}
