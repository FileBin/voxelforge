#version 450
layout(location = 0) in vec3 vColor;
layout(location = 1) in float vDepth;

layout(push_constant) uniform Model {
    mat4 m;
    vec4 tint;
} pc;

layout(set = 0, binding = 0) uniform Frame {
    mat4 vp;
    vec4 eye;
    vec4 light;
} fr;

layout(location = 0) out vec4 outColor;

void main() {
    // ближняя/дальняя планка отсечения линий — как в вьюере
    if (vDepth < fr.eye.w || vDepth > fr.light.w) discard;
    outColor = vec4(vColor, 1.0);
}
