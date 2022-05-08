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
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;


// -------------------------
// CUDA Related

// Params for the GPU
#define NUM_THREADS 256
int blks;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// // CUDA ErrorHandler
// inline void e(cudaError_t err, const char* file, int line){
//     if (err != cudaSuccess) 
//     {
//         printf("Error in %s at line %d:\n\t%s\n", file, line, cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }
// }

// #define HANDLE_ERROR(err) ( e(err, __FILE__, __LINE__) )
// -------------------------



struct Color{
    int r;
    int g;
    int b;
};

struct Vector2f{
    double x;
    double y;
};

struct Vertex{
    Vector2f position;
    Color color;
};

//Global constants
static const int point_num = 25000;
static const int steps_per_frame = 500;
static const double delta_per_step = 1e-5;
static const double t_start = -3.0;
static const double t_end = 3.0;
static std::mt19937 rand_gen;
static const int num_params = 18;

static const int intt_start = (int)(t_start/delta_per_step); // -3.0 / 1e-5 = -300000
static const int intt_end = (int)(t_end/delta_per_step); // 3.0 / 1e-5 = 300000
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

static int get_size(Vertex* vertex_array){
    return sizeof(vertex_array)/sizeof(vertex_array[0]);
}

static Color GetRandColor(int i) {
    i += 1;
    int r = std::min(255, 50 + (i * 11909) % 256);
    int g = std::min(255, 50 + (i * 52973) % 256);
    int b = std::min(255, 50 + (i * 44111) % 256);
    return Color{r, g, b};
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

__device__ void ToScreen(Vector2f& screenPt) {
  const float s = 0.25f * (double)window_w / 2.0;
  const float nx = (double)window_w * 0.5f + (float(screenPt.x) - 0.0) * s;
  const float ny = (double)window_h * 0.5f + (float(screenPt.y) - 0.0) * s;
  screenPt.x = nx;
  screenPt.y = ny;
}


// TODO: 參考一下compute each step裡面的stride的部分 是不是有可以加速的部分？
__global__ void compute_each_step(Vector2f* cuda_vector_array, double T) {
    // index
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;


    for (int step = id ; step < 1000; step = step + stride) //steps = 2000
    {
        double t = T + step * 1e-7;
        // bool isOffScreen = true;
        double x = t;
        double y = t;
        for (int iter = 0; iter < 800; ++iter) // 800
        {
            const double xx = x * x; const double yy = y * y; const double tt = t * t;
            const double xy = x * y; const double xt = x * t; const double yt = y * t;

            const double nx =   xx * 1 + yy * 0 + tt * 0 + 
                                xy * 0 + xt *-1 + yt * 1 + 
                                x  *-1 + y  * 0 + t  * 0 ;
            
            const double ny =   xx * 0 + yy *-1 + tt *-1 + 
                                xy *-1 + xt *-1 + yt *-1 + 
                                x  * 0 + y  *-1 + t  * 0 ;
            x = nx;
            y = ny;

            Vector2f screenPt;
            screenPt.x = x;
            screenPt.y = y;

            ToScreen(screenPt);
            if (iter < 100)
            {
                screenPt.x = FLT_MAX;
                screenPt.y = FLT_MAX;
            }

            cuda_vector_array[step*800 + iter].x = screenPt.x;
            cuda_vector_array[step*800 + iter].y = screenPt.y;

        } //iteration end
    } // step end
}

void gen_color(Vertex* vertex_array){
    for (int i = 0; i < get_size(vertex_array); ++i){
        vertex_array[i].color = GetRandColor(i % point_num);
    }
    return; 
}

__global__ void run_equation(Vertex* vertex_array, double* params){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 0*500 + (-300000) = 0 , 1*500+-300000 = -299995 
    int intt = (tid * steps_per_frame) + intt_start; 
    if (intt >= intt_end)
        return;

    double t = ((double)intt) * delta_per_step; // t = 2.99999 
    
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
        screenPt.x = x;
        screenPt.y = y;
        ToScreen(screenPt);
        
        x = nx;
        y = ny;

        vertex_array[tid*point_num + point].position = screenPt;  // 0*800 + point, 1*800+point...

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
    int num_vertex = frame_num * point_num;
    cout << "vertex num: " <<  num_vertex << endl;
    // Calculate CUDA Vars
    blks = (num_vertex + NUM_THREADS - 1) / NUM_THREADS; // blks calculation way?
    
    // CPU vertex arr
    Vertex* vertex_array;
    vertex_array = new Vertex[num_vertex]; // 1200 * 800
    gen_color(vertex_array); // TODO: Should parallelize this?

    // GPU vertex arr
    Vertex* gpu_vertex_array;
    cudaMalloc((void**)&gpu_vertex_array, num_vertex * sizeof(Vertex));
    cudaMemcpy(gpu_vertex_array, vertex_array, num_vertex * sizeof(Vertex), cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );
    
    // Gen Equation Params
    double params[num_params];              // 18 
    double gpu_params[num_params];              // 18 
    RandParams(params);
    cudaMalloc((void**)&gpu_params, num_params * sizeof(double));
    cudaMemcpy(gpu_params, params, num_params * sizeof(double), cudaMemcpyHostToDevice);

    run_equation<<<blks, NUM_THREADS>>>(gpu_vertex_array, gpu_params);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // cathc error from kernel synchronize
    // HANDLE_ERROR( cudaPeekAtLastError());
    // catch error from kernel asynchronize
    // HANDLE_ERROR( cudaDeviceSynchronize());

    cudaDeviceSynchronize();

    //     // Save state if necessary
    //     if (fsave.good() && (step % savefreq) == 0) {
    //         cudaMemcpy(parts, parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
    //         save(fsave, parts, num_parts, size);
    //     }
    // }
    // ------------------------
    
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds \n";
   
    return 0;
}
