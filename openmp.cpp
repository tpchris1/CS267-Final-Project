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
#include <png.h>
#include <omp.h>


using namespace std;

#define t_step 1e-3

//Global constants
static const int iters = 800;
static const int steps_per_frame = 500;
static const double delta_per_step = 1e-5;
static const double delta_minimum = 1e-7;
static const double t_start = -3.0;
static const double t_end = 3.0;
static const int fad_speed = 10;
static std::mt19937 rand_gen;
static const float dot_sizes[3] = { 1.0f, 3.0f, 10.0f };
static const int num_params = 18;

//Global variables
static int window_w = 1600;
static int window_h = 900;
static int window_bits = 24;
static float plot_scale = 0.25f;
static float plot_x = 0.0f;
static float plot_y = 0.0f;

int frame_num = 0;

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

static void ResetPlot() {
  plot_scale = 0.25f;
  plot_x = 0.0f;
  plot_y = 0.0f;
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

void create_png(vector<Vertex>& vertex_array, double t) {

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
	double file_name_double = (t + 3) * 1e+5;
	// cout << "filename: " << file_name_double << " ";

	char filename[40];
	// sprintf(filename , "./pic/%06d.png" , int(file_name_double));
	sprintf(filename, "./seq_pic/%06d.png", frame_num++);

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
	params[ 0] = 1; params[ 1] = 0; params[ 2] = 0;
	params[ 3] = 0; params[ 4] =-3; params[ 5] = 5;
	params[ 6] =-1; params[ 7] = 0; params[ 8] = 0;

	params[ 9] = 0; params[10] = -1; params[11] = -1;
	params[12] = 1; params[13] = 9; params[14] = -1;
	params[15] = 0; params[16] = -1; params[17] = 0;
}

void run_equation(double& t, double rolling_delta, double* params, vector<Vector2f>& history, vector<Vertex>& vertex_array){
    //Smooth out the stepping speed.
    // double t = *pt; // get the time 
    const double delta = delta_per_step; // 1e-5
    rolling_delta = rolling_delta*0.99 + delta*0.01; // ??

    for (int step = 0; step < steps_per_frame; ++step) // steps_per_frame = 500 多少步畫一張圖 
    {
        bool isOffScreen = true;
        double x = t;
        double y = t;
        cout << endl << "step: " << step << " t: " << t << " t_step: " << t_step <<" rd: " << rolling_delta << endl;

        for (int iter = 0; iter < iters; ++iter) // 800 產生幾個點
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

            // cout <<"N: " << iter << " x: " << x << " y: " << y << endl;

            Vector2f screenPt = ToScreen(x,y);
            // if (iter < 100) // iter太少->點太少，直接無視，渲染的時候顯得更好看，不會有很少的點出現->拔掉
            // {
            //     screenPt.x = FLT_MAX;
            //     screenPt.y = FLT_MAX;
            // }

            vertex_array[step*iters + iter].position = screenPt;

            // Check if dynamic delta should be adjusted
            if (screenPt.x > 0.0f && screenPt.y > 0.0f && screenPt.x < window_w && screenPt.y < window_h)
            {
                const float dx = history[iter].x - float(x);
                const float dy = history[iter].y - float(y);
                const double dist = double(500.0f * sqrt(dx*dx + dy*dy));
                rolling_delta = min(rolling_delta, max(delta / (dist + 1e-5), delta_minimum));
                isOffScreen = false;
                cout << "----not off----: " << t << ", "<< t_step << ", " << rolling_delta << endl;
            }

            history[iter].x = float(x);
            history[iter].y = float(y);

        } //iteration end

        if(isOffScreen){
            cout << "++++is off++++: " << t <<", "<< t_step << ", " << rolling_delta << endl;
        }
        // t += (isOffScreen) ? t_step : rolling_delta;
        t += rolling_delta;

        // *pt = t; // assign updated t back to orginal addr
        // t += t_step;
    } // step end
    return;
}

int main(int argc, char* argv[]) {
	clock_t start, stop;
	cout << "start computing........." << endl;
	start = clock();

	// const char* output = argv[2];

    //Set random seed
    rand_gen.seed((unsigned int)time(0));

    //Simulation variables
    double t = t_start;
    vector<Vector2f> history(iters);        //iters = 800
    double rolling_delta = delta_per_step;  // 1e-5
    double params[num_params];              // 18 

    // Setup the vertex array
    vector<Vertex> vertex_array(iters * steps_per_frame); // 800 * 500

    for (size_t i = 0; i < vertex_array.size(); ++i) 
        vertex_array[i].color = GetRandColor(i % iters);


    // Initialize random parameters
    ResetPlot();
    RandParams(params);

    while (true)
    {
        // Automatic restart
        if (t > t_end)
        {
			break;
            ResetPlot();
            RandParams(params);
        }

        run_equation(t, rolling_delta, params, history, vertex_array);

		// Draw the data
        // if (t > 0)
		create_png(vertex_array, t);
        cout << "time: " << t << endl;

    } // while end, t end
    // cout << "bug4" << endl;

	stop = clock();
	cout << double(stop - start) / CLOCKS_PER_SEC << endl;
    return 0;
}

