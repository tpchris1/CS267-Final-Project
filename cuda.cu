#include <iostream>
#include <random>
#include <sstream>
#include <cassert>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <string>
#include <cstring>
#include <ctime>
#include <chrono> 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;


// -------------------------
// CUDA Related

// Params for the GPU
#define NUM_THREADS 256
int blks;

// CUDA ErrorHandler
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




struct Color{
    int r;
    int g;
    int b;
};

typedef struct Vector2f{
    double x;
    double y;
}Vector2f;

struct Vertex{
    Vector2f position;
    Color color;
};

//Global constants
static const int point_num = 1000;
static const int steps_per_frame = 500;
static const double delta_per_step = 1e-5;
static const double t_start = -3.0;
static const double t_end = 3.0;
static std::mt19937 rand_gen;
static const int num_params = 18;

// static const int intt_start = (int)(t_start/delta_per_step); // -3.0 / 1e-5 = -300000
// static const int intt_end = (int)(t_end/delta_per_step); // 3.0 / 1e-5 = 300000
static const int intt_start = -300000; // -3.0 / 1e-5 = -300000
static const int intt_end = 300000; // 3.0 / 1e-5 = 300000
static const int frame_num = (intt_end - intt_start) / steps_per_frame;
static const bool isRender = false;

//Global variables
__device__ static int window_w = 1600;
__device__ static int window_h = 900;
static int window_bits = 24;
static float plot_scale = 0.25f;
static float plot_x = 0.0f;
static float plot_y = 0.0f;

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

static int ResetPlot() {
    plot_scale = 0.25f;
    plot_x = 0.0f;
    plot_y = 0.0f;
    return 0;
}

static void RandParams(double* params) {
    // std::uniform_int_distribution<int> rand_int(0, 3);
    // for (int i = 0; i < num_params; ++i) {
    //     const int r = rand_int(rand_gen);
    //     if (r == 0) {
    //         params[i] = 1.0f;
    //     } else if (r == 1) {
    //         params[i] = -1.0f;
    //     } else {
    //         params[i] = 0.0f;
    //     }
    // }
	params[ 0] = 1; params[ 1] = 0; params[ 2] = 0;
	params[ 3] = 0; params[ 4] =-1; params[ 5] = 1;
	params[ 6] =-1; params[ 7] = 0; params[ 8] = 0;

	params[ 9] = 0; params[10] =-1; params[11] =-1;
	params[12] =-1; params[13] =-1; params[14] =-1;
	params[15] = 0; params[16] =-1; params[17] = 0;
    
}

__device__ static Color GetRandColor(int i) {
    i += 1;
    int r = min(255, 50 + (i * 11909) % 256);
    int g = min(255, 50 + (i * 52973) % 256);
    int b = min(255, 50 + (i * 44111) % 256);
    return Color{r, g, b};
}

__global__ void gen_color(Vertex* vertex_array, int num_vertex){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_vertex)
        return;
    vertex_array[tid].color = GetRandColor(tid % point_num);
    return; 
}

// __device__ void ToScreen(Vector2f& screenPt) {
__device__ Vector2f ToScreen(double x, double y) {
    const double s = 0.25f * (double)window_w / 2.0;
    const double nx = (double)window_w * 0.5f + (x - 0.0) * s;
    const double ny = (double)window_h * 0.5f + (y - 0.0) * s;
    
    Vector2f screenPt;
    screenPt.x = nx;
    screenPt.y = ny;
    return screenPt;
}

__global__ void run_equation(Vertex* vertex_array){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid + intt_start >= intt_end)
        return;

    int intt = tid + intt_start; // 0 + (-300000) = -300000 , 1+-300000 = -299999  
    double t = ((double)intt) * delta_per_step; // t = 3.0, 2.99999 
    // printf("%d %d\n",tid,tid*point_num);

    double params[18];
    params[ 0] = 1; params[ 1] = 0; params[ 2] = 0;
	params[ 3] = 0; params[ 4] =-1; params[ 5] = 1;
	params[ 6] =-1; params[ 7] = 0; params[ 8] = 0;

	params[ 9] = 0; params[10] =-1; params[11] =-1;
	params[12] =-1; params[13] =-1; params[14] =-1;
	params[15] = 0; params[16] =-1; params[17] = 0;
    
    // int step = intt % 500; 
    // double t = ((double)(intt_start + frame * steps_per_frame + step) * delta_per_step); // -300000 + 0*500 + 10 * 0.00001->-2.9999
    // string pto = "frame: " + to_string(frame) + " step: " + to_string(step) + " t: " + to_string(t) + "\n";   
    // cout << pto;
    double x = t;
    double y = t;

    for (int point = 0; point < point_num; ++point) // 800 產生幾個點
    {
        const double xx = x * x; const double yy = y * y; const double tt = t * t;
        const double xy = x * y; const double xt = x * t; const double yt = y * t;

        const double nx = xx * params[ 0] + yy * params[ 1] + tt * params[ 2] + 
                    xy * params[ 3] + xt * params[ 4] + yt * params[ 5] + 
                    x  * params[ 6] + y  * params[ 7] + t  * params[ 8] ;
        
        const double ny = xx * params[ 9] + yy * params[10] + tt * params[11] + 
                    xy * params[12] + xt * params[13] + yt * params[14] + 
                    x  * params[15] + y  * params[16] + t  * params[17] ;
        Vector2f screenPt;
        screenPt = ToScreen(x,y);
        
        // printf("%d %d %d\n",tid, point, idx);
        vertex_array[tid*point_num + point].position = screenPt;  // 0*800 + point, 1*800+point...
        // __syncthreads();
        
        x = nx;
        y = ny;
          //synchronize the local threads writing to the local memory cache
    } //iteration end
    return;
}

int main(int argc, char* argv[]) {

    // Parse Args
    // if (find_arg_idx(argc, argv, "-h") >= 0) {
    //     std::cout << "Options:" << std::endl;
    //     std::cout << "-h: see this help" << std::endl;
    //     std::cout << "-n <int>: set number of particles" << std::endl;
    //     return 0;
    // }

    // Initialize Point Num
    // point_num = find_int_arg(argc, argv, "-n", 800); // 800 is default value

	cout << "start computing........." << endl;
    auto start_time = std::chrono::steady_clock::now();

    cout << "intt_start: " << intt_start << " intt_end: " << intt_end << " frame_num: " << frame_num << endl;
    cout << "point_num: " << point_num << endl; 

    // ------------------------
    // Actual Execution 

    // Vars
    int num_vertex = frame_num * steps_per_frame * point_num;
    cout << "vertex num: " << num_vertex << endl;
    // Calculate CUDA Vars
    blks = (num_vertex + NUM_THREADS - 1) / NUM_THREADS; // blks calculation way?
    
    // CPU vertex arr
    Vertex* vertex_array;
    vertex_array = new Vertex[num_vertex]; // 1200 * 800

    // GPU vertex arr
    Vertex* gpu_vertex_array;
    cudaMalloc((void**)&gpu_vertex_array, num_vertex * sizeof(Vertex));
    gpuErrchk( cudaPeekAtLastError() );
    
    // Gen Equation Params -> move to run equation
    // double params[num_params];              // 18 
    // double gpu_params[num_params];              // 18 
    // RandParams(params);
    // cudaMalloc((void**)&gpu_params, num_params * sizeof(double));
    // cudaMemcpy(gpu_params, params, num_params * sizeof(double), cudaMemcpyHostToDevice);

    gen_color<<<blks, NUM_THREADS>>>(gpu_vertex_array, num_vertex); 
    gpuErrchk( cudaDeviceSynchronize() );
    run_equation<<<blks, NUM_THREADS>>>(gpu_vertex_array);
    gpuErrchk( cudaDeviceSynchronize() );

    // cathc error from kernel synchronize
    // HANDLE_ERROR( cudaPeekAtLastError());
    // catch error from kernel asynchronize
    // HANDLE_ERROR( cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    cudaMemcpy(vertex_array, gpu_vertex_array, num_vertex * sizeof(Vertex), cudaMemcpyDeviceToHost);

    // Finiah Actual Execution 
    // ------------------------
    
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds \n";
   
    return 0;
}
