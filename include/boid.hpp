#ifndef BOID_HPP
#define BOID_HPP

#include <glm/glm.hpp> 
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp> 
#include <glm/gtc/random.hpp>     
#include <glm/gtc/constants.hpp> 
#include <cstdlib> 
#include <ctime>   
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct group_parameters {
    glm::vec3 direction;
    float bias_val = 0.0f;
};

class BoidSim
{
public:
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> velocities;
    std::vector<int> group_ids;

    float protected_range = 0.025f;
    float protected_range_squared = protected_range * protected_range;

    float visual_range = 65.0f;
    float visual_range_squared = visual_range * visual_range;

    float centering_factor = 0.005f;
    float matching_factor = 0.015f;
    float avoid_factor = 0.07f;

    float min_speed = 15.0f;
    float max_speed = 22.5f;

    float margin = 0.5f;
    float turnfactor = 1.5f;

    float max_bias = 0.03f;
    float bias_increment = 0.01f;

    int num_scout_groups = 0;
    std::unordered_map<int, group_parameters> group_bias;

    glm::vec3 box_min = glm::vec3(0.0f);
    glm::vec3 box_max = glm::vec3(50.0f);

    BoidSim(int num_scout_groups);
    ~BoidSim();

    void update(float dt);
    void keep_in_bounds(size_t i);
    void handle_scouts(size_t i);
    void init_boids(int num_boids, int max_scouts_per_group);
    void print_boids();
    glm::vec3 random_direction();
};

inline void BoidSim::update(float dt) {
    size_t count = positions.size();
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 pos_avg(0.0f), velo_avg(0.0f), close(0.0f);
        float neighbors = 0.0f;

        for (size_t j = 0; j < count; ++j) {
            if (i == j) continue;
            glm::vec3 offset = positions[i] - positions[j];
            float dist_sq = glm::length2(offset);

            if (dist_sq < visual_range_squared) {
                if (dist_sq < protected_range_squared)
                    close += offset;
                else {
                    pos_avg += positions[j];
                    velo_avg += velocities[j];
                    neighbors += 1.0f;
                }
            }
        }

        if (neighbors > 0.0f) {
            pos_avg /= neighbors;
            velo_avg /= neighbors;
            velocities[i] += (pos_avg - positions[i]) * centering_factor;
            velocities[i] += (velo_avg - velocities[i]) * matching_factor;
        }

        velocities[i] += close * avoid_factor;

        keep_in_bounds(i);
        handle_scouts(i);

        float speed = glm::length(velocities[i]);
        if (speed < min_speed)
            velocities[i] = glm::normalize(velocities[i]) * min_speed;
        else if (speed > max_speed)
            velocities[i] = glm::normalize(velocities[i]) * max_speed;

        positions[i] += velocities[i] * dt;
    }
}

inline void BoidSim::keep_in_bounds(size_t i) {
    for (int axis = 0; axis < 3; ++axis) {
        float& pos = positions[i][axis];
        float& vel = velocities[i][axis];
        float min = box_min[axis];
        float max = box_max[axis];

        if (pos < min + margin) {
            pos = min + margin;
            vel = std::abs(vel);
        } else if (pos > max - margin) {
            pos = max - margin;
            vel = -std::abs(vel);
        }
    }
}

inline void BoidSim::handle_scouts(size_t i) {
    int gid = group_ids[i];
    if (gid == 0) return;

    auto it = group_bias.find(gid);
    if (it != group_bias.end()) {
        group_parameters& params = it->second;
        float dot = glm::dot(velocities[i], params.direction);
        if (dot > 0.0f)
            params.bias_val = std::min(max_bias, params.bias_val + bias_increment);
        else
            params.bias_val = std::max(bias_increment, params.bias_val - bias_increment);

        velocities[i] = glm::normalize((1.0f - params.bias_val) * velocities[i] + params.bias_val * params.direction);
    }
}

inline void BoidSim::init_boids(int num_boids, int max_scouts_per_group) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    glm::vec3 center = (box_min + box_max) * 0.5f;
    glm::vec3 half_box = (box_max - box_min) * 0.25f;

    for (int g = 1; g <= num_scout_groups; ++g)
        group_bias[g] = group_parameters{glm::normalize(random_direction()), 0.0f};

    for (int i = 0; i < num_boids; ++i) {
        glm::vec3 pos = center + glm::vec3(
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f
        ) * half_box;

        int gid = 0;
        if (num_scout_groups > 0 && i < num_scout_groups * max_scouts_per_group)
            gid = (i / max_scouts_per_group) + 1;

        glm::vec3 dir = gid > 0 ? group_bias[gid].direction : random_direction();
        float speed = min_speed + static_cast<float>(rand()) / RAND_MAX * (max_speed - min_speed);

        positions.push_back(pos);
        velocities.push_back(dir * speed);
        group_ids.push_back(gid);
    }
}

inline glm::vec3 BoidSim::random_direction() {
    return glm::normalize(glm::sphericalRand(1.0f));
}

inline void BoidSim::print_boids() {
    for (size_t i = 0; i < positions.size(); ++i) {
        std::cout << "Boid " << i << ":\n"
                  << "  Position: (" << positions[i].x << ", " << positions[i].y << ", " << positions[i].z << ")\n"
                  << "  Velocity: (" << velocities[i].x << ", " << velocities[i].y << ", " << velocities[i].z << ")\n";
    }
}

inline BoidSim::BoidSim(int num_scout_groups) : num_scout_groups(num_scout_groups) {}

inline BoidSim::~BoidSim() {}

#endif // BOID_HPP
