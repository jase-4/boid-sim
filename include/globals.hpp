#ifndef GLOBALS_HPP
#define GLOBALS_HPP
#include "camera.hpp"



const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT =800;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

#endif