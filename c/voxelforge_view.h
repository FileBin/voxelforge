// Живое окно: SDL3 + Vulkan. Только рендер и ввод, вся логика — в Rust.
#pragma once
#include <stdint.h>

/* vf_status и vf_ctx живут в ядре: окно работает на девайсе ядра (шаг 0),
 * поэтому буферы тел видны и compute, и графике без копирования. */
#include "voxelforge_core.h"

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
    int key_space, key_r, key_t, key_f, key_f3;
    int key_w, key_a, key_s, key_d, key_q, key_e;
    int shift, ctrl;
} vf_input;

/* Счётчики сцены для строки под таблицей стадий: сколько тел реально
 * просчитывается в такте и сколько в нём точек контакта. Без них просадку
 * нельзя сопоставить с объёмом работы. */
typedef struct {
    uint32_t total;    /* всего тел в сцене (вместе со спящими) */
    uint32_t awake;    /* бодрствующих: только они входят в расчёт такта */
    uint32_t contacts; /* точек контакта в последнем такте окна */
} vf_scene_counts;

/* Строка оверлея диагностики: имя стадии и её время за последнюю секунду —
 * минимум, среднее и максимум по тактам окна. Строки копируются внутрь окна,
 * поэтому указатель нужен только на время вызова vf_view_perf. name — ASCII
 * (встроенный шрифт 5x7), с нулём. */
typedef struct {
    char name[24];
    float min_ms;
    float mean_ms;
    float max_ms;
    float share;
} vf_perf_row;

vf_status vf_view_new(vf_view **out, vf_ctx *core, int w, int h, const char *title,
                      const char *shader_dir);
void vf_view_free(vf_view *v);

/* Оверлей диагностики стадий: rows == NULL или n == 0 — скрыть. Вызывать каждый
 * кадр между vf_view_begin и vf_view_end; числа дублируются в консоль, здесь
 * они для глаза. tick_* — строка ИТОГО такта: мин/сред/макс за то же окно.
 * scene — счётчики тел и контактов, отдельной строкой под таблицей (NULL —
 * строку не рисовать). */
void vf_view_perf(vf_view *v, const vf_perf_row *rows, uint32_t n,
                  float tick_min_ms, float tick_mean_ms, float tick_max_ms,
                  const vf_scene_counts *scene);

// Опросить события. Возвращает 0, если окно закрыто.
int vf_view_poll(vf_view *v, vf_input *in);

// Камера на кадр: матрица view-projection (16 float, column-major) и позиция глаза.
void vf_view_camera(vf_view *v, const float *vp, const float *eye, float near_m, float far_m);

// Добавить геометрию. Возвращает id формы (>=0) или -1.
int32_t vf_view_add_mesh(vf_view *v, const float *pos, const float *nrm, uint32_t nv,
                         const uint32_t *idx, uint32_t ni);
int32_t vf_view_add_lines(vf_view *v, const float *pts, uint32_t n_lines);

// Буфер тел физики: поза читается шейдером прямо оттуда (тот же девайс).
void vf_view_set_bodies(vf_view *v, void *buffer, uint32_t count);

/* Две очереди (VF_SPLIT=1). tick_sem — таймлайн такта: кадр ждёт его значение
 * перед рисованием; frame_sem — таймлайн кадров: кадр сигналит его сам, чтобы
 * такт знал, когда банк прочитан. NULL/NULL — прежний путь (одна очередь). */
void vf_view_timelines(vf_view *v, void *tick_sem, void *frame_sem);
/* Перед vf_view_begin: банк `bank` опубликован тактом `value` — ждать его. */
void vf_view_tick_ready(vf_view *v, uint64_t value, uint32_t bank);

// Кадр. Поза не передаётся: body — номер тела в буфере, alpha — сглаживание
// между прошлым тактом и текущим.
void vf_view_begin(vf_view *v, float r, float g, float b);
void vf_view_draw(vf_view *v, int32_t mesh, uint32_t body, float alpha, const float *tint);
/* 0 — FIFO (показ привязан к развёртке), 1 — MAILBOX/IMMEDIATE. */
int vf_view_present_mode(vf_view *v);
void vf_view_draw_lines(vf_view *v, int32_t lines, const float *model, const float *tint);
void vf_view_end(vf_view *v);
