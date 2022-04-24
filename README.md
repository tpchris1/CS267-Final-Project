# CS267-Final-Project
## How to run
### OpenMP
1. Need to include `zlib` and `libpng` into Cori
2. Type `module load zlib`
3. Type `module load png`
4. Use `module list` to see if the alib and libpng is loaded
5. `mkdir build`
6. `cd build`
7. `cmake -DCMAKE_BUILD_TYPE=Release ..`
8. `make`
9. `salloc -N 1 -C knl -q interactive -t 01:00:00`
10. `./chaos`
