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
#include <png.h>
#include <omp.h>


using namespace std;

#define t_step 1e-3

//Global constants
static const int point_num = 800;
static const int steps_per_frame = 500;
static const double delta_per_step = 1e-5;
static const double delta_minimum = 1e-7;
static const double t_start = -3.0;
static const double t_end = 3.0;
static const int fad_speed = 10;
static std::mt19937 rand_gen;
static const float dot_sizes[3] = { 1.0f, 3.0f, 10.0f };
static const int num_params = 18;

static const int intt_start = (int)(t_start/delta_per_step); // -3.0 / 1e-5 = -300000
static const int intt_end = (int)(t_end/delta_per_step); // 3.0 / 1e-5 = 300000
static const int frame_num = (intt_end - intt_start) / steps_per_frame;
static const bool isRender = false;

//Global variables
static int window_w = 1600;
static int window_h = 900;
static int window_bits = 24;
static float plot_scale = 0.25f;
static float plot_x = 0.0f;
static float plot_y = 0.0f;

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
    Color  color;
};

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

static Vector2f ToScreen(double x, double y) {
  const float s = plot_scale * float(window_h / 2);  //0.25 * 900/2  = 112.5
  const float nx = float(window_w) * 0.5f + (float(x) - plot_x) * s;
  const float ny = float(window_h) * 0.5f + (float(y) - plot_y) * s;
  return Vector2f{nx, ny};
}

void write_png(const char* filename, const int width, const int height, const int* imageR, const int* imageG, const int* imageB) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); 	assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);										assert(info_ptr);

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);


    size_t row_size = 3 * width * sizeof(png_byte);

    png_bytep row = (png_bytep)malloc(row_size);

    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            png_bytep color = row + x * 3;
			color[0] = imageR[x + y * window_w];
			color[1] = imageG[x + y * window_w];
			color[2] = imageB[x + y * window_w];
        }
        png_write_row(png_ptr, row);
    }


    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void create_png(vector<Vertex>& vertex_array, int frame) {

	// allocate memory for image 
	size_t image_size = window_w * window_h * sizeof(int);
    int* imageR = (int*)malloc(image_size);
	int* imageG = (int*)malloc(image_size);
	int* imageB = (int*)malloc(image_size);
	memset(imageR, 0, image_size);
	memset(imageG, 0, image_size);
	memset(imageB, 0, image_size);

    // plot the points  
    for (size_t i = 0; i < vertex_array.size(); ++i) 
    {
        Vector2f screenPt = vertex_array[i].position; // double
        Color    color    = vertex_array[i].color; // int
		int x = int(screenPt.x);
		int y = int(screenPt.y);

        if (screenPt.x > 0.0f && screenPt.y > 0.0f && screenPt.x < window_w && screenPt.y < window_h)
        {
			imageR[x + y * window_w] = abs(imageR[x + y * window_w] - color.r);
			imageG[x + y * window_w] = abs(imageG[x + y * window_w] - color.g);
			imageB[x + y * window_w] = abs(imageB[x + y * window_w] - color.b);
        }
    }

    // start I/O
	// double file_name_double = (t + 3) * 1e+5; // t start with -3
	// cout << "filename: " << file_name_double << " ";

	char filename[40];
	// sprintf(filename , "./pic/%06d.png" , int(file_name_double));
	sprintf(filename, "./seq_pic/%06d.png", frame);

	// cout << filename << endl;
	write_png(filename, window_w, window_h, imageR, imageG, imageB);
	free(imageR);
	free(imageG);
	free(imageB);
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
    // 0:x^2
    // 1:y^2
    // 2:t^2
    // 3:xy
    // 4:xt
    // 5:yt
    // 6:x
    // 7:y
    // 8:t
	params[ 0] = 1; params[ 1] = 0; params[ 2] = 0;
	params[ 3] = 0; params[ 4] =-1; params[ 5] = 1;
	params[ 6] =-1; params[ 7] = 0; params[ 8] = 0;

	params[ 9] = 0; params[10] =-1; params[11] =-1;
	params[12] =-1; params[13] =-1; params[14] =-1;
	params[15] = 0; params[16] =-1; params[17] = 0;

	// params[ 0] = 2; params[ 1] = 0; params[ 2] = 0;
	// params[ 3] = 0; params[ 4] = 1; params[ 5] = 0;
	// params[ 6] = 1; params[ 7] = 0; params[ 8] = 0;

	// params[ 9] = 2; params[10] = -2; params[11] = -2;
	// params[12] = -1; params[13] = 0; params[14] = 1;
	// params[15] = -1; params[16] = 1; params[17] = 0;
}

void gen_color(vector<Vertex>& vertex_array){
    for (size_t i = 0; i < vertex_array.size(); ++i){
        vertex_array[i].color = GetRandColor(i % point_num);
    }
    return; 
}

void run_equation(int step, int frame, double* params, vector<Vertex>& vertex_array){
    // double t = ((double)intt) / 100000;
    // int step = intt % 500; 
    double t = ((double)(intt_start + frame * steps_per_frame + step) * delta_per_step); // -300000 + 0*500 + 10 * 0.00001->-2.9999
    // cout << "frame: " << frame << " step: " << step << " t: " << t << endl;   

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

        x = nx;
        y = ny;

        Vector2f screenPt = ToScreen(x,y);

        vertex_array[step*point_num + point].position = screenPt;

    } //iteration end
    return;
}

void run_one_frame(int frame, double* params, bool isRender){
    // Setup the vertex array and params
    vector<Vertex> vertex_array(point_num * steps_per_frame); // 800 * 500
    
    // Generate color for each point in vertex array
    // TODO: need to parallelize this 
    gen_color(vertex_array);
    
    // #pragma omp for 
    // run equation for steps_per_frame times
    for (int step = 0; step < steps_per_frame; step++){
        run_equation(step, frame, params, vertex_array);
    }

    // barrier 
    // TODO: do we need to put barrier here? 
    // #pragma omp barrier
        vertex_array.clear();
        vertex_array.shrink_to_fit();
    // render 
    // #pragma omp master
    // {
    //     // cout << "this is master\n" ;
    //     // create_png(vertex_array, frame);
    // }
    return;
}

int main(int argc, char* argv[]) {

    
	// clock_t start, stop;
	cout << "start computing........." << endl;
	// start = clock();
    auto start_time = std::chrono::steady_clock::now();

    //Set random seed
    rand_gen.seed(42);

    //Simulation variables
    vector<Vector2f> history(point_num);        //point_num = 800
    double params[num_params];              // 18 


    // int intt_start = (int)(t_start/delta_per_step); // -3.0 / 1e-5 = -300000
    // int intt_end = (int)(t_end/delta_per_step); // 3.0 / 1e-5 = 300000
    // int frame_num = (intt_end - intt_start) / steps_per_frame;
    // bool isRender = false;
    
    cout << "intt_start: " << intt_start << " intt_end: " << intt_end << " frame_num: " << frame_num << endl;

    // Initialize random parameters
    ResetPlot();
    RandParams(params);

    #pragma omp parallel default(shared)
    {
        // for(int intt=intt_start; intt<intt_end; intt++){}
        #pragma omp for
        for(int frame=0; frame<frame_num; frame++){
            cout << "frame: " << frame << endl;
            run_one_frame(frame, params, isRender);
        }
        // int step = 0;
        // int frame = 0;
        // // for(double t=t_start; t<=t_end; t+=delta_per_step){ // invalid iteration when make
        // for(int intt=-300000; intt<300000; intt++){
        //     cout << "time: " << intt << endl;

        //     run_equation(intt, params, history, vertex_array);
            
            // #pragma omp atomic
            // step++;

            // if(step == steps_per_frame)
            // {
            //     // create_png(vertex_array, t);
            //     #pragma omp atomic
            //     frame += 1;
            //     step = 0;
            //     // cout << "time: " << t << " frame: " << frame << " step:" << step << endl;
            // }
        // } 
    }

    ResetPlot();
    RandParams(params);

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds. \n";

    return 0;
}

