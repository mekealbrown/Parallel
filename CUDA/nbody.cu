#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define MAX_BODIES 8192
#define SOFTENING 0.1f
#define DAMPING 0.999f

// Structure for 3D vectors
typedef struct {
    float x, y, z;
} Vec3;

// Structure for body data
typedef struct {
    Vec3 position;
    Vec3 velocity;
    float mass;
} Body;

// Device arrays
Body *d_bodies;
float *d_positions; // For OpenGL VBO
struct cudaGraphicsResource *cuda_vbo_resource;

// Simulation parameters
int num_bodies = 1000;
float dt = 0.01f;
float G = 1.0f;
int running = 0;
int editing = 0; // 0: not editing, 1: num_bodies, 2: dt, 3: G
char input_buffer[32] = "";
int input_pos = 0;

// OpenGL variables
GLuint vbo, vao;
GLFWwindow *window;

// CUDA kernel to compute forces and update positions
__global__ void compute_forces(Body *bodies, int n, float dt, float G, float softening, float damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 force = {0.0f, 0.0f, 0.0f};
    Vec3 pos_i = bodies[i].position;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        Vec3 pos_j = bodies[j].position;
        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz + softening);
        float inv_dist3 = 1.0f / (dist * dist * dist);
        float force_mag = G * bodies[i].mass * bodies[j].mass * inv_dist3;
        force.x += force_mag * dx;
        force.y += force_mag * dy;
        force.z += force_mag * dz;
    }

    bodies[i].velocity.x = bodies[i].velocity.x * damping + (force.x / bodies[i].mass) * dt;
    bodies[i].velocity.y = bodies[i].velocity.y * damping + (force.y / bodies[i].mass) * dt;
    bodies[i].velocity.z = bodies[i].velocity.z * damping + (force.z / bodies[i].mass) * dt;

    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
}

// CUDA kernel to update VBO positions
__global__ void update_vbo(Body *bodies, float *positions, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    positions[3*i] = bodies[i].position.x;
    positions[3*i + 1] = bodies[i].position.y;
    positions[3*i + 2] = bodies[i].position.z;
}

// Initialize bodies with random positions and velocities
void init_bodies(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].position.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        bodies[i].position.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        bodies[i].position.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        bodies[i].velocity.x = 0.1f * (rand() / (float)RAND_MAX - 0.5f);
        bodies[i].velocity.y = 0.1f * (rand() / (float)RAND_MAX - 0.5f);
        bodies[i].velocity.z = 0.1f * (rand() / (float)RAND_MAX - 0.5f);
        bodies[i].mass = 1.0f;
    }
}

// OpenGL setup
void init_opengl() {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(1);
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_bodies * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    glPointSize(2.0f);
    glEnable(GL_DEPTH_TEST);
}

// Shader setup
GLuint create_shader_program() {
    const char *vertex_shader_src =
        "#version 330 core\n"
        "layout(location = 0) in vec3 aPos;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "    gl_Position = mvp * vec4(aPos, 1.0);\n"
        "}\n";

    const char *fragment_shader_src =
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "}\n";

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_src, NULL);
    glCompileShader(vertex_shader);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_src, NULL);
    glCompileShader(fragment_shader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

// Simple text rendering (minimal, using OpenGL quads)
void render_text(const char *text, float x, float y, float scale) {
    static const float char_width = 8.0f / WINDOW_WIDTH;
    static const float char_height = 16.0f / WINDOW_HEIGHT;
    glUseProgram(0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (int i = 0; text[i]; i++) {
        if (text[i] == '>') glColor3f(0.0f, 1.0f, 0.0f); // Highlight selected
        else glColor3f(1.0f, 1.0f, 1.0f);
        float cx = x + i * char_width * scale;
        float cy = y;
        glVertex2f(cx, cy);
        glVertex2f(cx + char_width * scale, cy);
        glVertex2f(cx + char_width * scale, cy + char_height * scale);
        glVertex2f(cx, cy + char_height * scale);
    }
    glEnd();
}

// Render menu
void render_menu() {
    char buf[128];
    snprintf(buf, sizeof(buf), "%sNum Bodies: %d", editing == 1 ? "> " : "  ", num_bodies);
    render_text(buf, 10.0f / WINDOW_WIDTH, 10.0f / WINDOW_HEIGHT, 1.0f);
    snprintf(buf, sizeof(buf), "%sTime Step: %.4f", editing == 2 ? "> " : "  ", dt);
    render_text(buf, 10.0f / WINDOW_WIDTH, 30.0f / WINDOW_HEIGHT, 1.0f);
    snprintf(buf, sizeof(buf), "%sGrav Constant: %.2f", editing == 3 ? "> " : "  ", G);
    render_text(buf, 10.0f / WINDOW_WIDTH, 50.0f / WINDOW_HEIGHT, 1.0f);
    snprintf(buf, sizeof(buf), "Input: %s", input_buffer);
    render_text(buf, 10.0f / WINDOW_WIDTH, 70.0f / WINDOW_HEIGHT, 1.0f);
    render_text("Press 1-3 to select, Enter to apply, Space to start", 10.0f / WINDOW_WIDTH, 90.0f / WINDOW_HEIGHT, 1.0f);
}

// Main rendering loop
void render(GLuint program, float time) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float cam_dist = 5.0f;
    float cam_x = cam_dist * sinf(time * 0.1f);
    float cam_z = cam_dist * cosf(time * 0.1f);
    float view[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        -cam_x, 0.0f, -cam_z, 1.0f
    };
    float proj[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.01f, -1.0f,
        0.0f, 0.0f, -0.2f, 0.0f
    };
    float mvp[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            mvp[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; k++)
                mvp[i*4 + j] += proj[i*4 + k] * view[k*4 + j];
        }

    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "mvp"), 1, GL_FALSE, mvp);

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, num_bodies);
    glBindVertexArray(0);

    render_menu();
}

// Key callback for parameter input
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_1) editing = 1, input_buffer[0] = '\0', input_pos = 0;
    else if (key == GLFW_KEY_2) editing = 2, input_buffer[0] = '\0', input_pos = 0;
    else if (key == GLFW_KEY_3) editing = 3, input_buffer[0] = '\0', input_pos = 0;
    else if (key == GLFW_KEY_SPACE) {
        running = 1;
        editing = 0;
        input_buffer[0] = '\0';
        input_pos = 0;
    }
    else if (editing && key == GLFW_KEY_ENTER) {
        if (input_pos > 0) {
            if (editing == 1) {
                int new_num = atoi(input_buffer);
                if (new_num > 0 && new_num <= MAX_BODIES) num_bodies = new_num;
            }
            else if (editing == 2) {
                float new_dt = atof(input_buffer);
                if (new_dt > 0.0f) dt = new_dt;
            }
            else if (editing == 3) {
                float new_G = atof(input_buffer);
                if (new_G > 0.0f) G = new_G;
            }
        }
        input_buffer[0] = '\0';
        input_pos = 0;
        editing = 0;
    }
    else if (editing && key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
        if (input_pos < 31) input_buffer[input_pos++] = '0' + (key - GLFW_KEY_0), input_buffer[input_pos] = '\0';
    }
    else if (editing && key == GLFW_KEY_PERIOD) {
        if (input_pos < 31) input_buffer[input_pos++] = '.', input_buffer[input_pos] = '\0';
    }
    else if (editing && key == GLFW_KEY_BACKSPACE && input_pos > 0) {
        input_buffer[--input_pos] = '\0';
    }
}

// Main function
int main() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "3D N-Body Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    init_opengl();
    GLuint shader_program = create_shader_program();

    CUDA_CHECK(cudaMalloc(&d_bodies, MAX_BODIES * sizeof(Body)));

    Body *h_bodies = (Body*)malloc(MAX_BODIES * sizeof(Body));
    init_bodies(h_bodies, num_bodies);
    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, num_bodies * sizeof(Body), cudaMemcpyHostToDevice));
    free(h_bodies);

    int threads_per_block = 256;
    int blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    float time = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (running) {
            compute_forces<<<blocks, threads_per_block>>>(d_bodies, num_bodies, dt, G, SOFTENING, DAMPING);
            CUDA_CHECK(cudaDeviceSynchronize());

            float *d_vbo;
            size_t size;
            CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &size, cuda_vbo_resource));
            update_vbo<<<blocks, threads_per_block>>>(d_bodies, d_vbo, num_bodies);
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

            if (running == 1) {
                running = 2;
                h_bodies = (Body*)malloc(MAX_BODIES * sizeof(Body));
                init_bodies(h_bodies, num_bodies);
                CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, num_bodies * sizeof(Body), cudaMemcpyHostToDevice));
                free(h_bodies);
                blocks = (num_bodies + threads_per_block - 1) / threads_per_block;
            }
        }

        render(shader_program, time);
        time += dt;

        glfwSwapBuffers(window);
    }

    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shader_program);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}