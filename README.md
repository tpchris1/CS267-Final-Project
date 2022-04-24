# CS267-Final-Project
## How to run
### OpenMP
- Need to load `zlib`, `libpng` and `ffmpeg` into Cori
- `module load zlib`
- `module load png`
- `module load ffmpeg`
- `module list` to see if the zlib and libpng is loaded
- `mkdir build`
- `cd build`
- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- `make`
- `salloc -N 1 -C knl -q interactive -t 01:00:00`
- `mkdir seq_pic` for picture output folder otherwise seg fault
- `./chaos`
- `cd seq_pic` 
- `ffmpeg -r 60 -f image2 -s 1600x900 -i %06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4` to output video

