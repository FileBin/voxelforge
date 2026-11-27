#ifndef VOXELFORGE_COMMON_GLSL
#define VOXELFORGE_COMMON_GLSL

/* Общие типы для всех compute-шейдеров физики.
 *
 * Раскладка тела намеренно «тупая»: девять vec4, 144 байта, выравнивание 16.
 * Иначе std430 начнёт подкладывать паддинги и Rust-сторона разъедется с GLSL.
 *
 * Сфера и коробка отличаются только полем half и кодом формы: у сферы
 * half = (r, r, r) и код 0, у коробки half — полуразмеры и код 1. Общая
 * формула «полуразмер вдоль нормали» тогда работает для обеих.
 */

struct Body {
    vec4 pos_r;       /* xyz — позиция,  w — радиус (для сферы)   */
    vec4 vel_m;       /* xyz — скорость, w — обратная масса        */
    vec4 ang_i;       /* xyz — угловая,  w — обр. момент инерции   */
    vec4 quat;        /* w, x, y, z                                */
    /* misc: x — таймер покоя, y — флаг сна, z/w — исходные обратные
     * масса и момент инерции (на время сна обнуляются, чтобы тело было
     * статикой для соседей, и восстанавливаются при пробуждении). */
    vec4 misc;
    /* anchor: xyz — позиция на прошлом снимке сна. Сон считается по
     * смещению за окно времени, а не по мгновенной скорости: медленно
     * ползущее тело тоже должно засыпать, иначе куча не встанет никогда. */
    vec4 anchor;
    /* half_shape: xyz — полуразмеры, w — код формы (0 сфера, 1 коробка). */
    vec4 half_shape;
    /* Поза на начало такта: по ней рендер интерполирует движение между
     * тактами 20 Гц и кадрами. Пишется проходом snapshot в начале такта. */
    vec4 prev_pos_r;
    vec4 prev_quat;
};

/* Контакт в списке тела: для самого тела нормаль смотрит НА партнёра. */
struct C {
    ivec4 meta;   /* x — индекс партнёра, yzw — зарезервировано */
    vec4 n_pen;   /* xyz — нормаль, w — проникновение           */
    vec4 point;   /* xyz — точка приложения импульса в мире     */
};

layout(push_constant) uniform Push {
    vec4 prm0;    /* dt, gravity, angular_damping, friction        */
    vec4 prm1;    /* beta, max_push, margin, cell_size             */
    ivec4 grid;   /* nx, ny, nz, n_bodies                          */
    vec4 lo;      /* ящик: минимум; w — порог энергии раскола, Дж    */
    vec4 hi;      /* ящик: максимум                                 */
} pc;

/* Ёмкость списка контактов на тело. 24 с запасом: у сферы радиуса r в
 * плотной упаковке 12 соседей, плюс пол и четыре стенки. */
const int CAP = 64;
/* Кольцо событий раскола: 64 тела — потолок такта, вдвое с запасом. */
const uint MAX_EVENTS = 128u;
const int GROUND = -1;

ivec3 cell_of(vec3 p, vec3 lo, float cell, ivec3 dim) {
    ivec3 c = ivec3(floor((p - lo) / cell));
    return clamp(c, ivec3(0), dim - 1);
}

int cell_index(ivec3 c, ivec3 dim) {
    return (c.z * dim.y + c.y) * dim.x + c.x;
}

/* Полуразмер вдоль направления: для оси-выровненных нормалей коробки это
 * точное расстояние от центра до грани, для сферы — радиус. */
float half_along(vec3 n, vec3 half_ext) {
    vec3 a = abs(n);
    return a.x * half_ext.x + a.y * half_ext.y + a.z * half_ext.z;
}

bool is_block(Body b) {
    return b.half_shape.w > 0.5;
}

/* Контакт двух тел. Нормаль возвращается в соглашении «от a к b»,
 * проникновение — положительное. false, если касания нет. */
bool pair_contact(Body a, Body b, out vec3 n_out, out float pen_out) {
    vec3 d = b.pos_r.xyz - a.pos_r.xyz;
    vec3 ha = a.half_shape.xyz;
    vec3 hb = b.half_shape.xyz;
    bool a_box = is_block(a);
    bool b_box = is_block(b);

    if (!a_box && !b_box) {
        float dist2 = dot(d, d);
        float rr = ha.x + hb.x;
        if (dist2 >= rr * rr || dist2 <= 1e-12) return false;
        float dist = sqrt(dist2);
        n_out = d / dist;
        pen_out = rr - dist;
        return true;
    }

    if (a_box && b_box) {
        /* Перекрытие по каждой оси; выталкиваем по оси наименьшего
         * перекрытия — это и есть нормаль коробочного контакта. */
        vec3 overlap = ha + hb - abs(d);
        if (overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0) return false;
        float pen = overlap.x;
        vec3 n = vec3(1.0, 0.0, 0.0) * sign(d.x);
        if (d.x == 0.0) n = vec3(1.0, 0.0, 0.0);
        if (overlap.y < pen) { pen = overlap.y; n = vec3(0.0, sign(d.y), 0.0); }
        if (overlap.z < pen) { pen = overlap.z; n = vec3(0.0, 0.0, sign(d.z)); }
        pen_out = pen;
        n_out = n;
        return true;
    }

    /* Смешанная пара: сфера против коробки. */
    bool sphere_is_a = !a_box;
    vec3 hs = sphere_is_a ? ha : hb;
    vec3 hbox = sphere_is_a ? hb : ha;
    vec3 rel = sphere_is_a ? -d : d;      /* от коробки к сфере */
    vec3 q = clamp(rel, -hbox, hbox);
    vec3 delta = rel - q;
    float dist = length(delta);
    float r = hs.x;
    if (dist >= r) return false;
    vec3 n_box_to_sphere = dist > 1e-6 ? delta / dist : vec3(0.0, 1.0, 0.0);
    n_out = sphere_is_a ? -n_box_to_sphere : n_box_to_sphere;
    pen_out = r - dist;
    return true;
}

#endif
