#version 450
layout(location = 0) in vec3 vN;
layout(location = 1) in vec3 vW;

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
    vec3 n = normalize(vN);
    vec3 l = normalize(fr.light.xyz);
    float d = max(dot(n, l), 0.0);
    float sky = 0.30 + 0.20 * n.y;
    vec3 base = pc.tint.rgb;
    vec3 col = base * (sky + 0.85 * d);
    vec3 v = normalize(fr.eye.xyz - vW);
    vec3 h = normalize(l + v);
    col += base * pow(max(dot(n, h), 0.0), 42.0) * 0.32;
    // подсветка контура: трещины и грани осколков читаются даже в тени
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    col += vec3(0.16, 0.20, 0.30) * rim * 0.6;
    outColor = vec4(col, 1.0);
}
