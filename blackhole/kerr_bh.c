#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <omp.h>
#include <immintrin.h> // AVX2

// Constants
#define WIDTH 1024
#define HEIGHT 768
#define G 6.67430e-11
#define C 299792458.0
#define M_SUN 1.989e30
#define AU 1.496e11 // Astronomical unit (m)

// Black hole parameters
float bh_mass = 4.0 * M_SUN;
float bh_spin = 0.9; // Spin parameter (0 to 1, near-maximal)
float rs, delta, sigma, a; // Kerr metric terms

// Simulation globals
float *framebuffer;
float time = 0.0;
double last_frame_time = 0.0;
float fps = 0.0;

// Kerr metric helpers
void update_kerr_params() {
    rs = 2.0 * G * bh_mass / (C * C); // Schwarzschild radius
    a = bh_spin * rs / 2.0; // Spin parameter in length units
}

// 4-vector struct for ray tracing
typedef struct { __m256d x, y, z, t; } Vec4;

// SIMD-optimized geodesic step (approximate Runge-Kutta 4)
void geodesic_step(Vec4 *pos, Vec4 *vel, float dt) {
    __m256d r2 = _mm256_add_pd(_mm256_mul_pd(pos->x, pos->x), _mm256_mul_pd(pos->y, pos->y));
    __m256d r = _mm256_sqrt_pd(r2);
    __m256d a2 = _mm256_set1_pd(a * a);
    __m256d rs_vec = _mm256_set1_pd(rs);
    __m256d rho2 = _mm256_add_pd(r2, a2);

    // Kerr metric terms (simplified, assumes equatorial plane for accretion disk)
    __m256d delta = _mm256_sub_pd(r2, _mm256_add_pd(_mm256_mul_pd(rs_vec, r), a2));
    __m256d omega = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(2.0 * G * bh_mass * a), r), _mm256_mul_pd(rho2, rho2));

    // Velocity update (basic relativistic terms)
    vel->x = _mm256_sub_pd(vel->x, _mm256_mul_pd(_mm256_div_pd(rs_vec, r2), vel->x));
    vel->y = _mm256_add_pd(vel->y, _mm256_mul_pd(omega, vel->t));

    // Position update
    pos->x = _mm256_add_pd(pos->x, _mm256_mul_pd(vel->x, _mm256_set1_pd(dt)));
    pos->y = _mm256_add_pd(pos->y, _mm256_mul_pd(vel->y, _mm256_set1_pd(dt)));
}

// Ray tracing with accretion disk
void trace_rays(float *buffer, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 4) { // AVX2: 4 doubles per vector
            Vec4 pos = { _mm256_set_pd(x+3, x+2, x+1, x), _mm256_set1_pd(y), _mm256_setzero_pd(), _mm256_setzero_pd() };
            Vec4 vel = { _mm256_set1_pd(-1.0), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_set1_pd(1.0) };

            // Normalize to [-10, 10] coordinate system
            pos.x = _mm256_mul_pd(_mm256_sub_pd(pos.x, _mm256_set1_pd(width / 2.0)), _mm256_set1_pd(20.0 / width));
            pos.y = _mm256_mul_pd(_mm256_sub_pd(pos.y, _mm256_set1_pd(height / 2.0)), _mm256_set1_pd(20.0 / height));

            float color[12] = {0}; // RGB for 4 pixels
            int hit_disk[4] = {0};
            float doppler[4] = {1.0};

            // Trace ray backwards
            for (int step = 0; step < 100; step++) {
                geodesic_step(&pos, &vel, 0.1);
                __m256d r = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(pos.x, pos.x), _mm256_mul_pd(pos.y, pos.y)));

                // Event horizon check
                __m256d horizon = _mm256_set1_pd(rs * (1.0 + sqrt(1.0 - bh_spin * bh_spin)) / 2.0);
                if (_mm256_movemask_pd(_mm256_cmp_pd(r, horizon, _CMP_LT_OQ))) break;

                // Accretion disk (equatorial plane, r = 3rs to 10rs)
                __m256d disk_inner = _mm256_set1_pd(3.0 * rs);
                __m256d disk_outer = _mm256_set1_pd(10.0 * rs);
                if (_mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(r, disk_inner, _CMP_GT_OQ),
                                                     _mm256_cmp_pd(r, disk_outer, _CMP_LT_OQ)))) {
                    double r_vals[4], v_vals[4];
                    _mm256_storeu_pd(r_vals, r);
                    _mm256_storeu_pd(v_vals, vel.y);
                    for (int i = 0; i < 4 && (x + i) < width; i++) {
                        if (r_vals[i] > disk_inner[0] && r_vals[i] < disk_outer[0]) {
                            hit_disk[i] = 1;
                            float v = v_vals[i] / C; // Orbital velocity approximation
                            doppler[i] = sqrt(1.0 - v * v) / (1.0 - v); // Relativistic Doppler
                            color[i*3] = 1.0 * doppler[i];     // Redshifted R
                            color[i*3 + 1] = 0.8 * doppler[i]; // G
                            color[i*3 + 2] = 0.5 / doppler[i]; // Blueshifted B
                        }
                    }
                    break;
                }
            }

            // Photon ring and background
            for (int i = 0; i < 4 && (x + i) < width; i++) {
                int idx = (y * width + (x + i)) * 3;
                if (!hit_disk[i]) {
                    float r = sqrt(pow((x + i - width / 2.0) * 20.0 / width, 2) +
                                  pow((y - height / 2.0) * 20.0 / height, 2));
                    if (r > 2.5 * rs && r < 3.0 * rs) { // Photon ring
                        buffer[idx] = buffer[idx + 1] = buffer[idx + 2] = 0.8;
                    } else {
                        buffer[idx] = buffer[idx + 1] = buffer[idx + 2] = 0.1; // Background
                    }
                } else {
                    buffer[idx] = color[i*3];
                    buffer[idx + 1] = color[i*3 + 1];
                    buffer[idx + 2] = color[i*3 + 2];
                }
            }
        }
    }
}

// OpenGL display
void display() {
    double current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    fps = 1.0 / (current_time - last_frame_time);
    last_frame_time = current_time;

    glClear(GL_COLOR_BUFFER_BIT);
    update_kerr_params();
    trace_rays(framebuffer, WIDTH, HEIGHT);

    glRasterPos2i(-1, -1);
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, framebuffer);

    // Data box
    char data[128];
    snprintf(data, sizeof(data), "Mass: %.2e kg\nSpin: %.2f\nRs: %.2e m\nFPS: %.1f",
             bh_mass, bh_spin, rs, fps);
    glColor3f(1.0, 1.0, 1.0);
    glRasterPos2f(-0.95, 0.85);
    for (char *c = data; *c != '\0'; c++) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
    }

    glutSwapBuffers();
}

void update(int value) {
    time += 0.05;
    bh_spin = 0.9 + 0.09 * sin(time); // Oscillate spin
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}

int main(int argc, char **argv) {
    framebuffer = (float *)malloc(WIDTH * HEIGHT * 3 * sizeof(float));
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Kerr Black Hole Simulation");

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glutDisplayFunc(display);
    glutTimerFunc(0, update, 0);

    glutMainLoop();
    free(framebuffer);
    return 0;
}