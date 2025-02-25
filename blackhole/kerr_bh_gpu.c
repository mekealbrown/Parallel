#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

// Window dimensions
#define WIDTH 1024
#define HEIGHT 768

// Black hole parameters
float bh_mass = 4.0 * 1.989e30;
float bh_spin = 0.9;
float time = 0.0;
float fps = 0.0;
double last_frame_time = 0.0;

// Shader program
GLuint shaderProgram;

// Load and compile shader
GLuint loadShader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        printf("Shader compilation error: %s\n", log);
    }
    return shader;
}

// Init shaders
void initShaders() {
    const char *vertexShaderSource = "#version 330 core\n"
        "layout(location = 0) in vec2 position;\n"
        "out vec2 fragCoord;\n"
        "void main() {\n"
        "    gl_Position = vec4(position, 0.0, 1.0);\n"
        "    fragCoord = position * vec2(1024.0, 768.0);\n"
        "}\n";

    const char *fragmentShaderSource = "#version 330 core\n"
        "in vec2 fragCoord;\n"
        "out vec4 fragColor;\n"
        "uniform float time;\n"
        "uniform float bh_mass;\n"
        "uniform float bh_spin;\n"
        "const float G = 6.67430e-11;\n"
        "const float C = 299792458.0;\n"
        "float rs() { return 2.0 * G * bh_mass / (C * C) * 1e6; }\n"
        "vec3 trace_ray(vec2 coord) {\n"
        "    vec2 uv = (fragCoord - vec2(512.0, 384.0)) / 384.0;\n" // Normalized [-1, 1]
        "    float r = length(uv);\n"
        "    float theta = atan(uv.y, uv.x);\n"
        "    float horizon = 0.2;\n"
        "    if (r < horizon) return vec3(0.3, 0.0, 0.0); // Bright red horizon\n"
        "    float disk_inner = 0.3;\n"
        "    float disk_outer = 1.0;\n"
        "    if (r > disk_inner && r < disk_outer) {\n"
        "        float v = sqrt(G * bh_mass / (r * 384.0 * 1e6)) / C;\n"
        "        float doppler = clamp(sqrt(1.0 - v * v) / (1.0 - v * cos(theta)), 0.5, 2.0);\n"
        "        vec3 color = vec3(1.0, 0.8, 0.5);\n"
        "        return color * doppler;\n"
        "    }\n"
        "    if (r > 0.25 && r < 0.35) return vec3(0.8, 0.8, 0.8); // Photon ring\n"
        "    return vec3(0.1, 0.1, 0.2); // Blue background\n"
        "}\n"
        "void main() {\n"
        "    vec3 color = trace_ray(fragCoord);\n"
        "    fragColor = vec4(color, 1.0);\n"
        "}\n";

    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        printf("Shader program linking error: %s\n", log);
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// Display function
void display() {
    double current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    fps = 1.0 / (current_time - last_frame_time);
    last_frame_time = current_time;

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);

    glUniform1f(glGetUniformLocation(shaderProgram, "time"), time);
    glUniform1f(glGetUniformLocation(shaderProgram, "bh_mass"), bh_mass);
    glUniform1f(glGetUniformLocation(shaderProgram, "bh_spin"), bh_spin);

    glBegin(GL_QUADS);
    glVertex2f(-1.0, -1.0);
    glVertex2f(1.0, -1.0);
    glVertex2f(1.0, 1.0);
    glVertex2f(-1.0, 1.0);
    glEnd();

    glColor3f(1.0, 1.0, 1.0); // White text
    glRasterPos2f(-0.9, 0.9); // Top-left, safe spot
    char data[128];
    float rs = 2.0 * 6.67430e-11 * bh_mass / (299792458.0 * 299792458.0) * 1e6;
    snprintf(data, sizeof(data), "Mass: %.2e kg\nSpin: %.2f\nRs: %.2e m\nFPS: %.1f",
             bh_mass, bh_spin, rs, fps);
    for (char *c = data; *c != '\0'; c++) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
    }

    glutSwapBuffers();
}

// Update function
void update(int value) {
    time += 0.05;
    bh_spin = 0.9 + 0.09 * sin(time);
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Kerr Black Hole Shader Simulation");

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("GLEW init failed: %s\n", glewGetErrorString(err));
        return 1;
    }

    printf("GLEW version: %s\n", glewGetString(GLEW_VERSION));
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));

    glClearColor(0.0, 0.0, 0.0, 1.0);
    initShaders();

    glutDisplayFunc(display);
    glutTimerFunc(0, update, 0);

    glutMainLoop();
    return 0;
}