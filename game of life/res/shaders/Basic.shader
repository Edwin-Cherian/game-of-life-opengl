#shader vertex
#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in float state;
out float v_State;
uniform mat4 u_MVP; // model view projection matrix

void main()
{
   gl_Position = u_MVP * position;
   v_State = state;
};



#shader fragment
#version 330 core

layout(location = 0) out vec4 color;
in float v_State;
uniform vec4 u_Color;

void main()
{

	color = vec4(0.05f, v_State, 0.05f, 1.0f);
};