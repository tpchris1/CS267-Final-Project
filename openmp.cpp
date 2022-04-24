#include "common.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    #pragma omp atomic
    particle.ax += coef * dx;
    #pragma omp atomic
    particle.ay += coef * dy;

    #pragma omp atomic
    neighbor.ax -= coef * dx;
    #pragma omp atomic
    neighbor.ay -= coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

typedef vector<particle_t*> bin_t;

double size;
bin_t* bins;
int bin_row_count;
int bin_count;
double bin_size;
omp_lock_t *lock;

int inline get_bin_id(particle_t& particle) {
    int x, y;
    x = particle.x / bin_size;
    y = particle.y / bin_size;
    if (x == bin_row_count) {
        x--;
    }
    if (y == bin_row_count) {
        y--;
    }
    return y * bin_row_count + x;
}

void reconstruct_bin(particle_t* parts, int num_parts) {
    #pragma omp for
    for (int i = 0; i < bin_count; i++) {
        bins[i].clear();
    }

    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
    	parts[i].ax = 0;
    	parts[i].ay = 0;
        int bin_id = get_bin_id(parts[i]);
        omp_set_lock(&(lock[bin_id]));
        bins[bin_id].push_back(&parts[i]);
        omp_unset_lock(&(lock[bin_id]));
    }
}


void init_simulation(particle_t* parts, int num_parts, double size_) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    bin_row_count = size_ / cutoff;
    bin_count = bin_row_count * bin_row_count;
    bin_size = size_ / bin_row_count;
    bins = new bin_t[bin_count];
    lock = new omp_lock_t[bin_count];

    #pragma omp parallel default(shared)
    {
        #pragma omp for
        for (int i=0; i< bin_count; i++)
            omp_init_lock(&(lock[i]));
        reconstruct_bin(parts, num_parts);
    }
}

void inline do_neighbor(int cur_bin_id, int i, int nei_bin_id) {
    for (particle_t* neighbor : bins[nei_bin_id]) {
        apply_force(*bins[cur_bin_id][i], *neighbor);
    }
}

void inline do_cur_bin(int i, int cur_bin_id) {
    if (i > 0){
        for (int j = i - 1; j > -1; --j){
            apply_force(*bins[cur_bin_id][i], *bins[cur_bin_id][j]);
        } 
    }
}

bool inline has_up(int bin_id) {
    return bin_id - bin_row_count > -1;
}
bool inline has_down(int bin_id) {
    return bin_id + bin_row_count < bin_count;
}
bool inline has_left(int bin_id) {
    return bin_id % bin_row_count != 0;
}
bool inline has_right(int bin_id) {
    return bin_id % bin_row_count != bin_row_count - 1;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces iterate through all the particles
    // each bin
    #pragma omp for  // TODO: also try 1
    for (int bin_id = 0; bin_id < bin_count; ++bin_id){
        // each particle in the bin
        
        for (int i = 0; i < bins[bin_id].size(); ++i){
            do_cur_bin(i, bin_id);
            // up
            if (has_up(bin_id)) {
                do_neighbor(bin_id, i, bin_id - bin_row_count);
            }
            // up right
            if (has_up(bin_id) && has_right(bin_id)) {
                do_neighbor(bin_id, i, bin_id - bin_row_count + 1);
            }
            // left
            if (has_left(bin_id)) {
                do_neighbor(bin_id, i, bin_id - 1);
            }
            // up left
            if (has_up(bin_id) && has_left(bin_id)) {
                do_neighbor(bin_id, i, bin_id - bin_row_count - 1);
            }
        }
    }

    // Move Particles
    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
    reconstruct_bin(parts, num_parts);
} 
