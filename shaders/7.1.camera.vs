#version 330 core
layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec2 aTexCoord;
layout (location = 3) in mat4 aInstanceMatrix;

//out vec2 TexCoord;
//out vec4 FragColor;

//uniform mat4 model;
//uniform vec3 objectColor;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * aInstanceMatrix * vec4(aPos, 1.0f);
	//FragColor = (objectColor,1.0);
	//TexCoord = aTexCoord;
}