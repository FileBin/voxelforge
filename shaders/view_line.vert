#version 450
layout(location = 0) in vec3 aPos;

layout(push_constant) uniform Model {
    mat4 m;
    vec4 tint;
} pc;

layout(set = 0, binding = 0) uniform Frame {
    mat4 vp;
    vec4 eye;
    vec4 light;
} fr;

layout(location = 0) out vec3 vColor;
layout(location = 1) out float vDepth;

void main() {
    vec4 w = pc.m * vec4(aPos, 1.0);
    vColor = pc.tint.rgb;
    vDepth = distance(w.xyz, fr.eye.xyz);
    gl_Position = fr.vp * w;
}
