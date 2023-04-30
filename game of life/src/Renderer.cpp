#include "Renderer.h"
#include <iostream>

void GLClearError()
{
    //here c++ loop pops an error off gl error stack if = 0 then loop ends
    //this just clears all the current errors from the error stack
    while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* function, const char* file, int line)
{
    //here c++ loop works by popping error and if it is 0 loop ends else it continues
    while (GLenum error = glGetError())
    {
        std::cout << "[OpenGL Error] (" << error << "):" << function << " " << file << ":" << line << std::endl;
        // if error occurs we return false so our assertion runs the debug break function
        return false;
    }
    return true;
}

void Renderer::Clear() const
{
    GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

void Renderer::Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader) const
{
    shader.Bind();
    va.Bind();
    GLCall(glDrawElements(GL_TRIANGLES, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
}