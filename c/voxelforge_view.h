// Живое окно: SDL3 + Vulkan. Только рендер и ввод, вся логика — в Rust.
#pragma once
#include <stdint.h>

typedef struct vf_status {
    int32_t code;
    const char *msg;
} vf_status;

typedef struct vf_view vf_view;

// Ввод за кадр.
typedef struct vf_input {
    int alive;      // 0 — окно закрыли
    int resized;
    uint32_t width, height;
    float mx, my;   // позиция мыши в пикселях
    float dx, dy;   // смещение с прошлого кадра
    float wheel;    // прокрутка за кадр
    int lmb, rmb;   // кнопки сейчас
    int lmb_pressed;    // нажатие в этом кадре
    int key_space, key_r, key_t, key_f;
    int key_w, key_a, key_s, key_d, key_q, key_e;
    int shift, ctrl;
} vf_input;

vf_status vf_view_new(int w, int h, const char *title, const char *shader_dir, vf_view **out);
void vf_view_free(vf_view *v);

// Опросить события. Возвращает 0, если окно закрыто.
int vf_view_poll(vf_view *v, vf_input *in);

// Камера на кадр: матрица view-projection (16 float, column-major) и позиция глаза.
void vf_view_camera(vf_view *v, const float *vp, const float *eye, float near_m, float far_m);

// Добавить геометрию. Возвращает id формы (>=0) или -1.
int32_t vf_view_add_mesh(vf_view *v, const float *pos, const float *nrm, uint32_t nv,
                         const uint32_t *idx, uint32_t ni);
int32_t vf_view_add_lines(vf_view *v, const float *pts, uint32_t n_lines);

// Кадр.
void vf_view_begin(vf_view *v, float r, float g, float b);
void vf_view_draw(vf_view *v, int32_t mesh, const float *model, const float *tint);
void vf_view_draw_lines(vf_view *v, int32_t lines, const float *model, const float *tint);
void vf_view_end(vf_view *v);
