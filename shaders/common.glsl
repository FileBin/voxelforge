#ifndef VOXELFORGE_COMMON_GLSL
#define VOXELFORGE_COMMON_GLSL

/* Общие типы для всех compute-шейдеров физики.
 *
 * Раскладка тел намеренно «тупая»: четыре vec4, 64 байта, выравнивание 16.
 * Иначе std430 начнёт подкладывать паддинги и Rust-сторона разъедется с GLSL.
 */

struct Body {
    vec4 pos_r;   /* xyz — позиция,   w — радиус           */
    vec4 vel_m;   /* xyz — скорость,  w — обратная масса   */
    vec4 ang_i;   /* xyz — угловая,   w — обр. момент инер. */
    vec4 quat;    /* w, x, y, z                            */
    /* misc: x — таймер покоя, y — флаг сна, z/w — исходные обратные
     * масса и момент инерции (на время сна обнуляются, чтобы тело было
     * статикой для соседей, и восстанавливаются при пробуждении). */
    vec4 misc;
};

/* Контакт в списке тела: для самого тела нормаль смотрит НА партнёра. */
struct C {
    ivec4 meta;   /* x — индекс партнёра, yzw — зарезервировано */
    vec4 n_pen;   /* xyz — нормаль, w — проникновение           */
};

layout(push_constant) uniform Push {
    vec4 prm0;    /* dt, gravity, angular_damping, friction        */
    vec4 prm1;    /* beta, max_push, margin, cell_size             */
    ivec4 grid;   /* nx, ny, nz, n_bodies                          */
    vec4 lo;      /* ящик: минимум                                  */
    vec4 hi;      /* ящик: максимум                                 */
} pc;

/* Ёмкость списка контактов на тело. 24 с запасом: у сферы радиуса r в
 * плотной упаковке 12 соседей, плюс пол и четыре стенки. */
const int CAP = 24;
const int GROUND = -1;

ivec3 cell_of(vec3 p, vec3 lo, float cell, ivec3 dim) {
    ivec3 c = ivec3(floor((p - lo) / cell));
    return clamp(c, ivec3(0), dim - 1);
}

int cell_index(ivec3 c, ivec3 dim) {
    return (c.z * dim.y + c.y) * dim.x + c.x;
}

#endif
