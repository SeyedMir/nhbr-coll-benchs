#ifndef BENCH_H
#define BENCH_H

#ifdef ERROR_CHECK
    #define Type_MPI MPI_INT
    typedef int Datatype;
#else
    #define Type_MPI MPI_CHAR
    typedef char Datatype;
#endif

#define LARGE_MSG_THR 1024
#define LARGE_MSG_ITR 300

#endif
