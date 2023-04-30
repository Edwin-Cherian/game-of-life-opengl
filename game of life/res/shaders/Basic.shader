#shader vertex
#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in float texOffset;

out vec2 v_TexCoord;
out float v_TexOffset;
uniform mat4 u_MVP; // model view projection matrix

void main()
{
   gl_Position = u_MVP * position;
   v_TexCoord = texCoord + texOffset * vec2(0.5, 0.0);
   v_TexOffset = texOffset;
};



#shader fragment
#version 330 core

layout(location = 0) out vec4 color;

uniform vec4 u_Color;
uniform sampler2D u_Textures;

in vec2 v_TexCoord;
in float v_TexOffset;

void main()
{
	vec4 texColor = texture(u_Textures, v_TexCoord);
	color = texColor;
	//color = vec4(0, (1.0f - 0), 0.0f, 1.0f);
};