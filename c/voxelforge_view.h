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

/* Времена кадра и такта за последнее окно (1 с) — строка над таблицей стадий
 * и два графика под ней. Считает главный поток: кадры он видит сам, работу
 * такта присылает физика.
 *
 * frame_ms — РАБОТА кадра (запись команд + submit), а не период: период при
 * 60 fps всегда 16.7 мс, его задаёт развёртка, и на графике он показывал бы
 * бюджет, а не нагрузку. tick_ms — «ИТОГО такт» (работа стадий), а не интервал
 * между тактами (50 мс при 20 Гц). fps/tps остались числом над графиками:
 * частота полезна, но её график — это график развёртки.
 * История — покадрово, старые отсчёты первыми; указатели живут только на время
 * вызова vf_view_perf (окно копирует данные внутрь себя). */
typedef struct {
    float fps_mean, fps_min, fps_max;
    float tps_mean, tps_min, tps_max;
    float frame_mean_ms, frame_min_ms, frame_max_ms; /* работа кадра, мс */
    float tick_mean_ms, tick_min_ms, tick_max_ms;    /* работа такта, мс */
    uint32_t frames;       /* кадров и тактов в окне; frames == 0 — окна ещё нет */
    uint32_t ticks;
    const float *frame_hist; /* работа кадра по кадрам окна, старые первыми */
    uint32_t frame_hist_n;
    const float *tick_hist;  /* работа последнего такта на каждый кадр */
    uint32_t tick_hist_n;
} vf_rates;

vf_status vf_view_new(vf_view **out, vf_ctx *core, int w, int h, const char *title,
                      const char *shader_dir);
void vf_view_free(vf_view *v);

/* Оверлей диагностики стадий: rows == NULL или n == 0 — скрыть. Вызывать каждый
 * кадр между vf_view_begin и vf_view_end; числа дублируются в консоль, здесь
 * они для глаза. tick_* — строка ИТОГО такта: мин/сред/макс за то же окно.
 * scene — счётчики тел и контактов, отдельной строкой под таблицей (NULL —
 * строку не рисовать). rates — частоты кадра и физики: строка над стадиями и
 * два графика под ней (NULL — строку и графики не рисовать). */
void vf_view_perf(vf_view *v, const vf_perf_row *rows, uint32_t n,
                  float tick_min_ms, float tick_mean_ms, float tick_max_ms,
                  const vf_scene_counts *scene, const vf_rates *rates);

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

// Кадр. Поза не передаётся: body — номер тела в буфере, alpha — сглаживание
// между прошлым тактом и текущим.
void vf_view_begin(vf_view *v, float r, float g, float b);
void vf_view_draw(vf_view *v, int32_t mesh, uint32_t body, float alpha, const float *tint);
/* 0 — FIFO (показ привязан к развёртке), 1 — MAILBOX/IMMEDIATE. */
int vf_view_present_mode(vf_view *v);
void vf_view_draw_lines(vf_view *v, int32_t lines, const float *model, const float *tint);
void vf_view_end(vf_view *v);
/* Реальная работа последнего кадра, мс: запись команд и submit (CPU-часть
 * кадра). Ожидание fence/acquire, пауза развёртки и сам present не входят —
 * на 60 Гц они всегда равны периоду 16.7 мс и о нагрузке не говорят ничего.
 * 0 — кадр ещё не измерен. */
float vf_view_frame_work_ms(const vf_view *v);
