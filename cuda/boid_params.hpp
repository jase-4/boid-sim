#ifndef BOID_PARAMS_HPP
#define BOID_PARAMS_HPP

struct BoidParams {
    float protected_range_sq;
    float visual_range_sq;
    float centering_factor;
    float matching_factor;
    float avoid_factor;
    float min_speed;
    float max_speed;
    float margin;
    float bias_increment;
};


#endif