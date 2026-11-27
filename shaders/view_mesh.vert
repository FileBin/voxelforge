#version 450
/* Поза тела берётся прямо из буфера физики: CPU больше не присылает матрицу
 * и не считает сглаживание. pick.x — номер тела, pick.y — alpha между прошлым
 * тактом и текущим (сглаживание движения теперь на устройстве). */
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;

layout(push_constant) uniform Pick {
    vec4 tint;
    vec4 pick;
} pc;

layout(set = 0, binding = 0) uniform Frame {
    mat4 vp;
    vec4 eye;
    vec4 light;
} fr;

/* Раскладка Body из common.glsl: 9 vec4 (pos_r, vel_m, ang_i, quat, misc,
 * anchor, half_shape, prev_pos_r, prev_quat). */
layout(std430, set = 0, binding = 1) buffer Bodies { vec4 b[]; } bodies;

layout(location = 0) out vec3 vN;
layout(location = 1) out vec3 vW;

const int STRIDE = 9;

vec3 qrot(vec4 q, vec3 v) {
    vec3 u = q.yzw;
    float w = q.x;
    vec3 t = 2.0 * cross(u, v);
    return v + w * t + cross(u, t);
}

mat4 model_of(uint bi, float alpha) {
    int o = int(bi) * STRIDE;
    vec3 cur = bodies.b[o + 0].xyz;
    vec4 qc = bodies.b[o + 3];
    vec3 prev = bodies.b[o + 7].xyz;
    vec4 qp = bodies.b[o + 8];
    vec3 p = mix(prev, cur, alpha);
    /* Кватернионы q и -q — одно вращение: без выравнивания знака nlerp идёт
     * «длинной дорогой» и тело дёргается на развороте. */
    if (dot(qp, qc) < 0.0) qp = -qp;
    vec4 q = normalize(mix(qp, qc, alpha));
    mat4 m = mat4(1.0);
    m[0].xyz = qrot(q, vec3(1.0, 0.0, 0.0));
    m[1].xyz = qrot(q, vec3(0.0, 1.0, 0.0));
    m[2].xyz = qrot(q, vec3(0.0, 0.0, 1.0));
    m[3] = vec4(p, 1.0);
    return m;
}

void main() {
    mat4 m = model_of(uint(pc.pick.x), pc.pick.y);
    vec4 w = m * vec4(aPos, 1.0);
    vW = w.xyz;
    vN = mat3(m) * aNrm;
    gl_Position = fr.vp * w;
}
