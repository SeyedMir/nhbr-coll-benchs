Set of benchmarks to evaluate the performance of MPI neighborhood collectives

Developed these as part of my work on design and implementation of optimized
communication pattern and schedules for MPI neighborhood collectives.

- [Related PR in MPICH](https://github.com/pmodels/mpich/pull/3892)
- [Related paper](https://www.mcs.anl.gov/~balaji/pubs/2017/hipc/hipc17.nhbrcoll.pdf)

## Build

```
make
```

## Usage

### `./run_bench.sh -n 16 rsg-nhbr-allgather`
This will run the benckmark with 16 processes and will use a
random sparse graph for the process neighborhood topology.

## TODO

- Add alltoall benchmarks
