#include <string>
#include <iostream>
#include <random>
#include <ctime>
#include <array>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <sstream>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "VertexBufferLayout.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Texture.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

#include "Solver.h"
#include <vector>
#include <typeinfo>
#include "Grid.h"
#include <chrono>

//#include "test.h"
#include "gtest.h"

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    float WINDOW_WIDTH  = 800.0f;
    float WINDOW_HEIGHT = 800.0f;

    window = glfwCreateWindow((int) WINDOW_WIDTH, (int) WINDOW_HEIGHT, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }



    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    //sets the swap interval for the current OpenGL or OpenGL ES context, 
    //i.e. the number of screen updates to wait from the time glfwSwapBuffers 
    //was called before swapping the buffers and returning.  AKA vsync
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK)
        std::cout << "ERROR: could not initialise glew" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;
    
    // <------ define constants and settings ------>
    srand((unsigned int) time(NULL));
    const int vertex_acount = 3;
    const int quad_vacount = 4 * vertex_acount;
    const int quad_icount = 6;

    const float p_size = 19.0f;
    const float spacing = 1.0f;
    const int cols = (int) WINDOW_WIDTH  / (p_size + spacing);
    const int rows = (int) WINDOW_HEIGHT / (p_size + spacing);
    const int cells = rows * cols;
    float* positions = new float[cells * quad_vacount];
    cudaMallocManaged(&positions, cells * quad_vacount * sizeof(float));
    float* positions_buffer = new float[cells * quad_vacount];
    cudaMallocManaged(&positions_buffer, cells * quad_vacount * sizeof(float));
    unsigned int* indices = new unsigned int[cells * 6];
    


    Grid* grid = new Grid;



    // fill position array with vertex buffer data for particles
    float x_offset = 1.0;
    float y_offset = 1.0;
    for (int i = 0; i < rows; i++)
    {   
        for (int j = 0; j < cols; j++) {
            // top left
            positions[(i * cols + j) * quad_vacount +  0] = x_offset;
            positions[(i * cols + j) * quad_vacount +  1] = y_offset;
            positions[(i * cols + j) * quad_vacount +  2] = 0.0f;
            // top rig(h * cols + j)t  quad_vacount     
            positions[(i * cols + j) * quad_vacount +  3] = x_offset + p_size;
            positions[(i * cols + j) * quad_vacount +  4] = y_offset;
            positions[(i * cols + j) * quad_vacount +  5] = 0.0f;
            // bottom (  * cols + j)   quad_vacount    
            positions[(i * cols + j) * quad_vacount +  6] = x_offset + p_size;
            positions[(i * cols + j) * quad_vacount +  7] = y_offset + p_size;
            positions[(i * cols + j) * quad_vacount +  8] = 0.0f;
            // bottom (  * cols + j)   quad_vacount     
            positions[(i * cols + j) * quad_vacount +  9] = x_offset;
            positions[(i * cols + j) * quad_vacount + 10] = y_offset + p_size;
            positions[(i * cols + j) * quad_vacount + 11] = 0.0f;

            x_offset += p_size + spacing;

            //grid->AddObject(&positions[i * quad_vacount]);
        }
        x_offset = 1.0;
        y_offset += p_size + spacing;
    }

    x_offset = 1.0;
    y_offset = 1.0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) {
            // top left
            positions_buffer[(i * cols + j) * quad_vacount + 0] = x_offset;
            positions_buffer[(i * cols + j) * quad_vacount + 1] = y_offset;
            positions_buffer[(i * cols + j) * quad_vacount + 2] = 0.0f;
            // top ri_bufferg(h * cols + j)t  quad_vacount     
            positions_buffer[(i * cols + j) * quad_vacount + 3] = x_offset + p_size;
            positions_buffer[(i * cols + j) * quad_vacount + 4] = y_offset;
            positions_buffer[(i * cols + j) * quad_vacount + 5] = 0.0f;
            // bottom_buffer (  * cols + j)   quad_vacount    
            positions_buffer[(i * cols + j) * quad_vacount + 6] = x_offset + p_size;
            positions_buffer[(i * cols + j) * quad_vacount + 7] = y_offset + p_size;
            positions_buffer[(i * cols + j) * quad_vacount + 8] = 0.0f;
            // bottom_buffer (  * cols + j)   quad_vacount     
            positions_buffer[(i * cols + j) * quad_vacount + 9] = x_offset;
            positions_buffer[(i * cols + j) * quad_vacount + 10] = y_offset + p_size;
            positions_buffer[(i * cols + j) * quad_vacount + 11] = 0.0f;

            x_offset += p_size + spacing;

            //grid->AddObject(&positions[i * quad_vacount]);
        }
        x_offset = 1.0;
        y_offset += p_size + spacing;
    }



    for (int i = -1; i < 2; i++)
    {
        positions[((i+2) * cols + 2) * quad_vacount +  2] = 1.0f;
        positions[((i+2) * cols + 2) * quad_vacount +  5] = 1.0f;
        positions[((i+2) * cols + 2) * quad_vacount +  8] = 1.0f;
        positions[((i+2) * cols + 2) * quad_vacount + 11] = 1.0f;
    }
    /*for (int i = -1; i < 2; i++)
    {
        positions[(30 * cols + (i+30)) * quad_vacount + 2] = 1.0f;
        positions[(30 * cols + (i+30)) * quad_vacount + 5] = 1.0f;
        positions[(30 * cols + (i+30)) * quad_vacount + 8] = 1.0f;
        positions[(30 * cols + (i+30)) * quad_vacount + 11] = 1.0f;
    }*/
    


    // fill indices array with index buffer data for particles
    for (int i = 0; i < cells; i++)
    {
        // join order: ( TL, TR, BR )
        indices[i * quad_icount + 0] = i * 4;
        indices[i * quad_icount + 1] = i * 4 + 1;
        indices[i * quad_icount + 2] = i * 4 + 2;
        // join order: ( BR, BL, TL )
        indices[i * quad_icount + 3] = i * 4 + 2;
        indices[i * quad_icount + 4] = i * 4 + 3;
        indices[i * quad_icount + 5] = i * 4;
    }


    
    // brackets to create a scope and its just because opengl is annoying
    // and won't "properly" terminate otherwise
    {
        GLCall(glEnable(GL_BLEND))
        GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        // vao = vertex array object
        unsigned int vao;
        GLCall(glGenVertexArrays(1, &vao));
        GLCall(glBindVertexArray(vao));

        VertexArray va;
        VertexBuffer vb(positions,  cells * quad_vacount * 4); // cells * vertices attributes per quad * 4 for size of 1 attribute
        VertexBufferLayout layout;
        layout.Push<float>(2); // 2 floats for position
        layout.Push<float>(1); // 1 float for alive/dead

        va.AddBuffer(vb, layout);

        IndexBuffer ib(indices, cells * 6); // cells = # quads and each quad = 6 indices

        // projection matrix
        glm::vec3 translation(0, 0, 0);
        glm::mat4 proj = glm::ortho(0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, -1.0f, 1.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0));
        glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
        glm::mat4 mvp = proj * view * model;


        // setting up shaders
        Shader shader("res/shaders/Basic.shader");
        shader.Bind();
        shader.SetUniform4f("u_Color", 0.2f, 0.9f, 0.4f, 1.0f);
        shader.SetUniformMat4f("u_MVP", mvp);
        

        Renderer renderer;
        Solver solver(WINDOW_WIDTH, WINDOW_HEIGHT, rows, cols, quad_vacount);

        // initialise gui
        ImGui::CreateContext();
        ImGui_ImplGlfwGL3_Init(window, true);
        ImGui::StyleColorsDark();

        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.90f, 1.00f);

        // initialise r value for rect colour
        float r = 0.1f;
        float increment = 0.01f;

        double xpos, ypos;
        bool running = true;

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            if (running)
                solver.updateCells(positions, positions_buffer);

            // keyboard event handlers
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                running = false;
            }
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                running = true;
            }
            if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
                solver.resetCells(positions);
            }

            // mouse event handlers
            if (glfwGetWindowAttrib(window, GLFW_HOVERED) && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
                glfwGetCursorPos(window, &xpos, &ypos);
                //ypos = row and xpos = col
                solver.setCellState(positions, (int)(ypos / (p_size + spacing)), (int)(xpos / (p_size + spacing)), 1.0f);
            } 
            else if (glfwGetWindowAttrib(window, GLFW_HOVERED) && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)) {
                glfwGetCursorPos(window, &xpos, &ypos);
                //ypos = row and xpos = col
                solver.setCellState(positions, (int)(ypos / (p_size + spacing)), (int)(xpos / (p_size + spacing)), 0.0f);
            }

            
            
            vb.UpdateBuffer(positions, cells * quad_vacount * 4);

            /* Render here */
            renderer.Clear();

            ImGui_ImplGlfwGL3_NewFrame();

            shader.Bind();
            shader.SetUniform4f("u_Color", r, 0.2f, 0.3f, 1.0f);
            shader.SetUniformMat4f("u_MVP", mvp);

            renderer.Draw(va, ib, shader);
            

            if (r > 1.0f || r < 0.0f)
                increment *= -1;

            r += increment;

            /* Render gui */
            {
                ImGui::Text("Settings");
                ImGui::SliderFloat3("Translation", &translation.x, -200.0f, 200.0f);
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            }
            model = glm::translate(glm::mat4(1.0f), translation);
            mvp = proj * view * model;

            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
          
        }
    }

    cudaFree(positions);
    //delete[] positions;
    delete[] indices;
    grid->~Grid();
    ImGui_ImplGlfwGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}

