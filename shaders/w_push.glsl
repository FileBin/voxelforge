#ifndef VF_WATER_PUSH
#define VF_WATER_PUSH

layout(push_constant) uniform WPush {
    vec4 prm0;   /* dt, h (радиус сглаживания), rho0, stiffness */
    vec4 prm1;   /* k_near, sigma(viscosity), floor_y, beta     */
    vec4 prm2;   /* nbuckets, cap, 0, 0                         */
} wp;

const uint NBUCKETS = 32768u;
const uint CAP = 12u;

uint hash_cell(ivec3 c) {
    uint h = uint(c.x) * 92837111u ^ uint(c.y) * 689287499u ^ uint(c.z) * 283923481u;
    return h % NBUCKETS;
}

ivec3 cell_of(vec3 p, float h) {
    return ivec3(floor(p / h));
}

#endif
