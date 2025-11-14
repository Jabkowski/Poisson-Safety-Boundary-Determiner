# Poisson-Safety-Boundary-Determiner
Poisson Safety Boundary Determiner - ml based supervised model

## Artificial Data Generation
Directory *matlab* contains scripts which:
- generates set of maps with randomly located circular obstacles (obstacles
can be separated or occluded),
- generates based on them Poisson Safety Function with derivatives over *x*
and *y*,
- saves results to **.h5* file.

This scripts can be run directly in malab or there can be used Docker file for
run it with MCR (MATLAB Compiler Runtime) - with compiled version of script, so
no MATLAB license is needed.

### Docker
To run generation with MCR you can use docker with follwing commands:
#### Build
```console
docker build -t mcr_data_generation .
```
#### Run
Run with 2 arguments:
- output file name,
- number of grids generated.
```console
docker run --rm mcr_data_generation output_file_name.h5 1000
```


