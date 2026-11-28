//! Диагностика производительности: агрегация GPU-таймстемпов по стадиям такта.
//!
//! Ядро (`c/voxelforge_core.c`) пишет по метке до каждого диспатча и после
//! последнего, а результаты отдаёт на СЛЕДУЮЩЕМ такте — внутри такта ничего не
//! ждётся. Здесь из этих длительностей собирается окно в одну секунду:
//! минимум, среднее, максимум и доля от такта на каждую стадию, плюс
//! неровномерность диспетчей внутри стадии (макс/среднее) — она показывает
//! перекос нагрузки, который средние скрывают.
//!
//! Минимум и максимум важны не меньше среднего: среднее прячет и всплеск
//! (стадия изредка берёт втрое больше), и «пустой» такт. Три числа рядом —
//! разброс стадии за окно.
//!
//! Рядом со стадиями — частоты: сколько кадров в секунду и сколько тактов
//! физики реально завершилось (`Rates`), с мин/макс за то же окно и историей
//! на графики. Номинал тут бесполезен: такт «20 Гц» при перегрузе выходит
//! 12 Гц, и именно эта разница и есть ответ на «почему лагает».
//!
//! Печать — одна таблица раз в секунду, самой дорогой стадией сверху: вопрос
//! «что оптимизировать» должен решаться чтением первой строки.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::sim::PIPE_NAMES;

/// Сколько стадий вообще может прийти: пайплайнов сейчас `PIPE_NAMES.len()`.
const MAX_STAGES: usize = 64;

/// Длина истории для графиков: 120 кадров ≈ 2 с при 60 fps.
pub const HIST_MAX: usize = 120;

#[derive(Clone, Copy)]
struct StageAcc {
    /// Сумма времени стадии за такт — по всем тактам окна.
    tick_sum_ms: f64,
    /// Минимум времени стадии за один такт окна. Стартует с +INF: 0.0 значило
    /// бы «стадия однажды отработала мгновенно» и портило бы минимум.
    tick_min_ms: f32,
    /// Максимум времени стадии за один такт.
    tick_max_ms: f32,
    /// Сумма длительностей отдельных диспатчей (для среднего по диспетчам).
    disp_sum_ms: f64,
    /// Самый долгий отдельный диспатч стадии.
    disp_max_ms: f32,
    /// Сколько раз стадия встречалась в окне (диспатчей).
    disp_n: u32,
}

impl Default for StageAcc {
    fn default() -> Self {
        Self {
            tick_sum_ms: 0.0,
            tick_min_ms: f32::INFINITY,
            tick_max_ms: 0.0,
            disp_sum_ms: 0.0,
            disp_max_ms: 0.0,
            disp_n: 0,
        }
    }
}

/// Одна строка таблицы — то, что печатается и отдаётся оверлею.
#[derive(Clone, Copy)]
pub struct PerfRow {
    pub stage: usize,
    /// Минимум времени стадии за такт в окне, мс. Считается по тактам, где
    /// стадия правда работала: условный диспатч не тянет минимум к нулю.
    pub min_ms: f32,
    pub mean_ms: f32,
    pub max_ms: f32,
    /// Доля среднего времени стадии в среднем такте, %.
    pub share: f32,
    /// Максимум/среднее по диспетчам внутри стадии (1.0 — ровно).
    pub imbal: f32,
}

/// Строка для оверлея в окне. Раскладка 1:1 с `vf_perf_row` в
/// `c/voxelforge_view.h`: имя — ASCII в фиксированном буфере, чтобы не
/// передавать через границу указатели на Rust-строки.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VfPerfRow {
    pub name: [u8; 24],
    pub min_ms: f32,
    pub mean_ms: f32,
    pub max_ms: f32,
    pub share: f32,
}

/// Счётчики сцены: сколько тел реально просчитывается в такте. Печатаются
/// строкой под таблицей стадий (и дублируются в оверлей) — просадку надо
/// сопоставлять не только со стадией, но и с объёмом работы.
/// Раскладка 1:1 с `vf_scene_counts` в `c/voxelforge_view.h`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VfSceneCounts {
    /// Действующих тел в сцене — вместе со спящими.
    pub total: u32,
    /// Бодрствующих: только они входят в расчёт такта.
    pub awake: u32,
    /// Точек контакта в последнем такте окна (сумма списков контактов).
    pub contacts: u32,
}

/// Итог окна частот: сколько кадров и тактов пришлось на секунду окна.
/// Печатается строкой над таблицей стадий и рисуется в оверлее.
/// Раскладка 1:1 с `vf_rates` в `c/voxelforge_view.h`.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct RatesOut {
    pub fps_mean: f32,
    /// Крайние значения мгновенного FPS в окне (по худшему/лучшему кадру).
    pub fps_min: f32,
    pub fps_max: f32,
    pub tps_mean: f32,
    /// Крайние значения TPS в окне (по кадрам окна).
    pub tps_min: f32,
    pub tps_max: f32,
    /// Среднее время кадра, мс: по нему видно, упираемся ли в развёртку.
    pub frame_mean_ms: f32,
    /// Кадров и тактов в окне. `frames == 0` — окно ещё ни разу не закрылось.
    pub frames: u32,
    pub ticks: u32,
}

/// `RatesOut` + история для графиков, для оверлея в окне. Указатели живут
/// только на время вызова `vf_view_perf`: окно копирует данные внутрь себя.
/// Раскладка 1:1 с `vf_rates` в `c/voxelforge_view.h`.
#[repr(C)]
pub struct VfRatesView {
    pub fps_mean: f32,
    pub fps_min: f32,
    pub fps_max: f32,
    pub tps_mean: f32,
    pub tps_min: f32,
    pub tps_max: f32,
    pub frame_mean_ms: f32,
    pub frames: u32,
    pub ticks: u32,
    /// Покадровая история, старые отсчёты первыми.
    pub fps_hist: *const f32,
    pub fps_hist_n: u32,
    pub tps_hist: *const f32,
    pub tps_hist_n: u32,
}

/// Счётчик частот окна: кадры считает главный поток, такты приходят числом
/// завершённых тактов физики (`Perf::ticks_total`). Окно — то же 1 с.
///
/// TPS здесь — не номинал: это число тактов, которые правда доехали до конца
/// за последнюю секунду, и минимум с максимумом по кадрам окна. Голодный такт
/// (физика не успевает) виден как просадка графика, а не как «20 Гц» на бумаге.
pub struct Rates {
    /// Сколько секунд копить до закрытия окна (1.0 в бою).
    pub window_s: f32,
    start: Instant,
    frames: u32,
    ticks: u32,
    frame_sum_ms: f64,
    frame_min_ms: f32,
    frame_max_ms: f32,
    tps_min: f32,
    tps_max: f32,
    tps_any: bool,
    /// Моменты завершённых тактов за последнюю секунду — мгновенный TPS.
    pubs: VecDeque<Instant>,
    hist_fps: [f32; HIST_MAX],
    hist_tps: [f32; HIST_MAX],
    hist_n: u32,
    out: RatesOut,
    /// Первые кадры — создание окна и компиляция пайплайнов, а не рендер.
    /// Без пропуска первый же кадр в сотни мс растянул бы шкалу графиков.
    skip: u32,
}

impl Rates {
    pub fn new() -> Self {
        Self {
            window_s: 1.0,
            start: Instant::now(),
            frames: 0,
            ticks: 0,
            frame_sum_ms: 0.0,
            frame_min_ms: f32::INFINITY,
            frame_max_ms: 0.0,
            tps_min: 0.0,
            tps_max: 0.0,
            tps_any: false,
            pubs: VecDeque::new(),
            hist_fps: [0.0; HIST_MAX],
            hist_tps: [0.0; HIST_MAX],
            hist_n: 0,
            out: RatesOut::default(),
            skip: 3,
        }
    }

    /// Один кадр: `frame_ms` — настоящее (не урезанное клампом) время кадра,
    /// `ticks` — сколько тактов физики завершилось с прошлого кадра.
    /// Возвращает `true`, если этим кадром закрылось окно.
    pub fn sample(&mut self, now: Instant, frame_ms: f32, ticks: u32) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            self.start = now;
            return false;
        }
        let frame_ms = frame_ms.max(0.001);
        /* Мгновенный TPS — число тактов за последнюю секунду. Считаем по
         * моментам публикаций: на 20 Гц это 20 вёдер в скользящем окне, и
         * график выходит ровным, а голодный такт сразу топит значение. */
        for _ in 0..ticks.min(8) {
            self.pubs.push_back(now);
        }
        while let Some(&t) = self.pubs.front() {
            if now.duration_since(t).as_secs_f32() > 1.0 {
                self.pubs.pop_front();
            } else {
                break;
            }
        }
        let tps_now = self.pubs.len() as f32;
        if self.tps_any {
            self.tps_min = self.tps_min.min(tps_now);
            self.tps_max = self.tps_max.max(tps_now);
        } else {
            self.tps_min = tps_now;
            self.tps_max = tps_now;
            self.tps_any = true;
        }
        self.frames += 1;
        self.ticks += ticks;
        self.frame_sum_ms += frame_ms as f64;
        self.frame_min_ms = self.frame_min_ms.min(frame_ms);
        self.frame_max_ms = self.frame_max_ms.max(frame_ms);
        self.push_hist(1000.0 / frame_ms, tps_now);
        if now.duration_since(self.start).as_secs_f32() < self.window_s {
            return false;
        }
        self.close(now);
        true
    }

    fn close(&mut self, now: Instant) {
        let span = now.duration_since(self.start).as_secs_f32().max(1e-4);
        let frames = self.frames.max(1) as f32;
        let fps_mean = self.frames as f32 / span;
        let tps_mean = self.ticks as f32 / span;
        /* Один кадр в окне (софтверный рендер, долгий стопор) — крайние
         * значения вышли бы за среднее просто из-за границ окна: окно длиннее
         * суммы кадров. Тогда честнее показать среднее вместо «min > mean».
         * От двух кадров крайние уже имеют смысл. */
        let few = self.frames < 2;
        let fps_min = if few || !(self.frame_max_ms > 0.0) {
            fps_mean
        } else {
            1000.0 / self.frame_max_ms
        };
        let fps_max = if few || !(self.frame_min_ms.is_finite() && self.frame_min_ms > 0.0) {
            fps_mean
        } else {
            1000.0 / self.frame_min_ms
        };
        self.out = RatesOut {
            fps_mean,
            /* Мгновенный FPS по крайним кадрам окна: среднее прячет один
             * 40-мс кадр среди пятидесяти 16-мс — а глаз видит именно его. */
            fps_min,
            fps_max,
            tps_mean,
            tps_min: if self.tps_any && !few { self.tps_min } else { tps_mean },
            tps_max: if self.tps_any && !few { self.tps_max } else { tps_mean },
            frame_mean_ms: (self.frame_sum_ms / frames as f64) as f32,
            frames: self.frames,
            ticks: self.ticks,
        };
        self.frames = 0;
        self.ticks = 0;
        self.frame_sum_ms = 0.0;
        self.frame_min_ms = f32::INFINITY;
        self.frame_max_ms = 0.0;
        self.tps_min = 0.0;
        self.tps_max = 0.0;
        self.tps_any = false;
        self.start = now;
    }

    fn push_hist(&mut self, fps: f32, tps: f32) {
        let n = self.hist_n as usize;
        if n < HIST_MAX {
            self.hist_fps[n] = fps;
            self.hist_tps[n] = tps;
            self.hist_n += 1;
        } else {
            self.hist_fps[..].copy_within(1.., 0);
            self.hist_tps[..].copy_within(1.., 0);
            self.hist_fps[HIST_MAX - 1] = fps;
            self.hist_tps[HIST_MAX - 1] = tps;
        }
    }

    /// Итог последнего закрытого окна.
    pub fn out(&self) -> RatesOut {
        self.out
    }

    /// Итог + история для оверлея окна.
    pub fn c_view(&self) -> VfRatesView {
        VfRatesView {
            fps_mean: self.out.fps_mean,
            fps_min: self.out.fps_min,
            fps_max: self.out.fps_max,
            tps_mean: self.out.tps_mean,
            tps_min: self.out.tps_min,
            tps_max: self.out.tps_max,
            frame_mean_ms: self.out.frame_mean_ms,
            frames: self.out.frames,
            ticks: self.out.ticks,
            fps_hist: self.hist_fps.as_ptr(),
            fps_hist_n: self.hist_n,
            tps_hist: self.hist_tps.as_ptr(),
            tps_hist_n: self.hist_n,
        }
    }

    /// Начать заново (включение сбора, сброс мира).
    pub fn restart(&mut self) {
        self.start = Instant::now();
        self.frames = 0;
        self.ticks = 0;
        self.frame_sum_ms = 0.0;
        self.frame_min_ms = f32::INFINITY;
        self.frame_max_ms = 0.0;
        self.tps_min = 0.0;
        self.tps_max = 0.0;
        self.tps_any = false;
        self.pubs.clear();
        self.hist_n = 0;
        self.out = RatesOut::default();
        self.skip = 3;
    }
}

impl Default for Rates {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Perf {
    /// Драйвер умеет таймстемпы на этой очереди.
    pub supported: bool,
    /// Сбор включён (F3 / VF_PERF=1).
    pub enabled: bool,
    /// Итог окна частот (кадры/такты) — печатается первой строкой. Его считает
    /// главный поток (кадры) и присылает сюда перед тактом: одна таблица
    /// печатается одним местом, поэтому строка fps/tps всегда над стадиями.
    rates: RatesOut,
    /// Сколько тактов завершилось с включения сбора — главному потоку для TPS.
    ticks_total: u64,
    /// Счётчики сцены (тела, контакты) — берутся из готовых буферов ядра.
    scene: VfSceneCounts,
    acc: [StageAcc; MAX_STAGES],
    /// Сколько тактов попало в окно.
    ticks: u32,
    tick_sum_ms: f64,
    /// Минимум/максимум длительности такта за окно (+INF до первого такта).
    tick_min_ms: f32,
    tick_max_ms: f32,
    /// С какого момента копится окно.
    window_start: Instant,
    /// Готовые строки последнего окна — для оверлея в окне.
    pub rows: Vec<PerfRow>,
    pub tick_min_window_ms: f32,
    pub tick_mean_ms: f32,
    pub tick_max_window_ms: f32,
    /// Сколько секунд копить до печати (1.0 в бою; в тестах подменяется).
    pub window_s: f32,
    /// Сколько первых тактов пропустить: первый такт после включения — это
    /// компиляция пайплайнов драйвером, а не физика. Без пропуска самая
    /// дорогая стадия в первой таблице — всегда та, что первой компилируется.
    skip: u32,
}

impl Perf {
    pub fn new(supported: bool, enabled: bool) -> Self {
        Self {
            supported,
            enabled,
            rates: RatesOut::default(),
            ticks_total: 0,
            scene: VfSceneCounts::default(),
            acc: [StageAcc::default(); MAX_STAGES],
            ticks: 0,
            tick_sum_ms: 0.0,
            tick_min_ms: f32::INFINITY,
            tick_max_ms: 0.0,
            window_start: Instant::now(),
            rows: Vec::new(),
            tick_min_window_ms: 0.0,
            tick_mean_ms: 0.0,
            tick_max_window_ms: 0.0,
            window_s: 1.0,
            skip: 1,
        }
    }

    /// Один закрытый такт: длительности диспатчей и их номера стадий.
    /// `tick_ms` — от первой метки до последней.
    pub fn record(&mut self, stages: &[u32], ms: &[f32], tick_ms: f32) {
        if !self.enabled {
            return;
        }
        /* Такт доехал до конца — это и есть TPS. Считаем и тот первый такт,
         * что ниже пропускается из статистики стадий: иначе первая секунда
         * показывала бы физику без одного такта. */
        self.ticks_total += 1;
        if stages.is_empty() {
            return;
        }
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        let mut sum = [0.0f32; MAX_STAGES];
        let mut mx = [0.0f32; MAX_STAGES];
        let mut cnt = [0u32; MAX_STAGES];
        for (k, &s) in stages.iter().enumerate() {
            let s = s as usize;
            if s >= MAX_STAGES {
                continue;
            }
            let d = ms.get(k).copied().unwrap_or(0.0).max(0.0);
            sum[s] += d;
            if d > mx[s] {
                mx[s] = d;
            }
            cnt[s] += 1;
        }
        for (s, a) in self.acc.iter_mut().enumerate() {
            if cnt[s] == 0 {
                continue;
            }
            a.tick_sum_ms += sum[s] as f64;
            if sum[s] < a.tick_min_ms {
                a.tick_min_ms = sum[s];
            }
            if sum[s] > a.tick_max_ms {
                a.tick_max_ms = sum[s];
            }
            a.disp_sum_ms += sum[s] as f64;
            if mx[s] > a.disp_max_ms {
                a.disp_max_ms = mx[s];
            }
            a.disp_n += cnt[s];
        }
        self.ticks += 1;
        let tick_ms = tick_ms.max(0.0);
        self.tick_sum_ms += tick_ms as f64;
        if tick_ms < self.tick_min_ms {
            self.tick_min_ms = tick_ms;
        }
        if tick_ms > self.tick_max_ms {
            self.tick_max_ms = tick_ms;
        }
    }

    /// Пора печатать? Если да — считает окно, обновляет `rows` и сбрасывает
    /// накопители. Возвращает `true`, когда таблица действительно напечатана.
    pub fn maybe_print(&mut self, now: Instant) -> bool {
        if !self.enabled || self.ticks == 0 {
            return false;
        }
        if now.duration_since(self.window_start).as_secs_f32() < self.window_s {
            return false;
        }
        self.finish_window();
        self.print_table();
        self.window_start = now;
        true
    }

    /// Окно готово закрыться (но ещё не закрыто). Нужен вызывающему, чтобы
    /// прочитать счётчики сцены РОВНО на границе окна: раз в секунду, а не в
    /// каждом такте — иначе диагностика сама начинает тормозить такт.
    pub fn window_due(&self, now: Instant) -> bool {
        self.enabled
            && self.ticks > 0
            && now.duration_since(self.window_start).as_secs_f32() >= self.window_s
    }

    /// Счётчики сцены (тела, контакты) — из готовых буферов, не из чтения тел.
    pub fn set_scene(&mut self, total: u32, awake: u32, contacts: u32) {
        self.scene = VfSceneCounts {
            total,
            awake: awake.min(total),
            contacts,
        };
    }

    /// Счётчики сцены для оверлея (копия — данные переживают границу FFI).
    pub fn scene_counts(&self) -> VfSceneCounts {
        self.scene
    }

    /// Итог окна частот от главного потока: печатается строкой над стадиями.
    pub fn set_rates(&mut self, r: RatesOut) {
        self.rates = r;
    }

    /// Сколько тактов завершилось с включения сбора (монотонно).
    pub fn ticks_total(&self) -> u64 {
        self.ticks_total
    }

    /// Закрыть окно без печати (тесты и оверлей).
    pub fn finish_window(&mut self) {
        let ticks = self.ticks.max(1) as f32;
        let tick_mean = (self.tick_sum_ms / ticks as f64) as f32;
        let mut rows: Vec<PerfRow> = (0..MAX_STAGES)
            .filter(|&s| self.acc[s].disp_n > 0)
            .map(|s| {
                let a = &self.acc[s];
                let mean = (a.tick_sum_ms / ticks as f64) as f32;
                let disp_mean = (a.disp_sum_ms / a.disp_n.max(1) as f64) as f32;
                PerfRow {
                    stage: s,
                    /* Минимум мог остаться +INF, только если стадия не попала
                     * ни в один такт, — такие строки отфильтрованы выше. */
                    min_ms: if a.tick_min_ms.is_finite() { a.tick_min_ms } else { mean },
                    mean_ms: mean,
                    max_ms: a.tick_max_ms,
                    share: if tick_mean > 0.0 { 100.0 * mean / tick_mean } else { 0.0 },
                    imbal: if disp_mean > 1e-9 { a.disp_max_ms / disp_mean } else { 1.0 },
                }
            })
            .collect();
        rows.sort_by(|a, b| b.mean_ms.total_cmp(&a.mean_ms));
        self.rows = rows;
        self.tick_min_window_ms = if self.tick_min_ms.is_finite() {
            self.tick_min_ms
        } else {
            tick_mean
        };
        self.tick_mean_ms = tick_mean;
        self.tick_max_window_ms = self.tick_max_ms;
        self.acc = [StageAcc::default(); MAX_STAGES];
        self.ticks = 0;
        self.tick_sum_ms = 0.0;
        self.tick_min_ms = f32::INFINITY;
        self.tick_max_ms = 0.0;
    }

    fn print_table(&self) {
        /* Строка частот — первой: «сколько кадров и тактов» волнует раньше,
         * чем раскладка такта по стадиям. Кадры считает главный поток и
         * присылает сюда, так что строка и таблица — один печатный заход. */
        let r = self.rates;
        if r.frames > 0 {
            /* Ниже 10 — с десятой: в headless/на просадках целые «2 fps» и
             * «0 fps» скрывают разницу между 0.3 и 9. */
            let f = |v: f32| if v >= 10.0 { format!("{v:.0}") } else { format!("{v:.1}") };
            println!(
                "[perf] fps {} (min {} max {}) · tps {} (min {} max {}) · frame {:.1} мс",
                f(r.fps_mean),
                f(r.fps_min),
                f(r.fps_max),
                f(r.tps_mean),
                f(r.tps_min),
                f(r.tps_max),
                r.frame_mean_ms
            );
        } else {
            println!("[perf] fps — (окно кадров ещё не закрыто) · tps —");
        }
        println!(
            "[perf] такт: мин {:.2} · среднее {:.2} · макс {:.2} мс · стадии по убыванию",
            self.tick_min_window_ms, self.tick_mean_ms, self.tick_max_window_ms
        );
        println!(
            "  {:<18} {:>8} {:>10} {:>9} {:>6} {:>7}",
            "стадия", "мин", "среднее", "макс", "доля%", "дисп"
        );
        for r in &self.rows {
            let name = PIPE_NAMES.get(r.stage).copied().unwrap_or("?");
            println!(
                "  {:<18} {:>8.3} {:>10.3} {:>9.3} {:>5.1}% {:>7.2}",
                name, r.min_ms, r.mean_ms, r.max_ms, r.share, r.imbal
            );
        }
        println!(
            "  {:<18} {:>8.3} {:>10.3} {:>9.3} {:>5.1}%",
            "ИТОГО такт", self.tick_min_window_ms, self.tick_mean_ms, self.tick_max_window_ms, 100.0
        );
        /* Строка объёма работы: просадка стадии без числа тел ничего не
         * объясняет — 30 мс на 13 телах и на 60 телах это разные вещи. */
        println!(
            "  тел {} (бодр {}, спят {}) · контактов {}",
            self.scene.total,
            self.scene.awake,
            self.scene.total.saturating_sub(self.scene.awake),
            self.scene.contacts
        );
    }

    /// Начать окно заново (включение сбора с чистого листа).
    pub fn restart_window(&mut self) {
        self.window_start = Instant::now();
        self.acc = [StageAcc::default(); MAX_STAGES];
        self.ticks = 0;
        self.tick_sum_ms = 0.0;
        self.tick_min_ms = f32::INFINITY;
        self.tick_max_ms = 0.0;
        self.tick_min_window_ms = 0.0;
        self.tick_mean_ms = 0.0;
        self.tick_max_window_ms = 0.0;
        self.scene = VfSceneCounts::default();
        self.rows.clear();
        self.skip = 1;
    }

    /// Период окна для тестов: пока окно не закрыто, печати нет.
    pub fn window_elapsed(&self, now: Instant) -> Duration {
        now.duration_since(self.window_start)
    }

    /// Среднее время такта в последнем закрытом окне, мс.
    pub fn tick_ms(&self) -> f32 {
        self.tick_mean_ms
    }

    /// Минимум/среднее/максимум такта последнего закрытого окна, мс.
    pub fn tick_range_ms(&self) -> (f32, f32, f32) {
        (
            self.tick_min_window_ms,
            self.tick_mean_ms,
            self.tick_max_window_ms,
        )
    }

    /// Строки для оверлея в окне (копия — данные переживают границу FFI).
    pub fn overlay_rows(&self) -> Vec<VfPerfRow> {
        self.rows
            .iter()
            .map(|r| {
                let mut name = [0u8; 24];
                let s = PIPE_NAMES.get(r.stage).copied().unwrap_or("?");
                let n = s.len().min(23);
                name[..n].copy_from_slice(&s.as_bytes()[..n]);
                VfPerfRow {
                    name,
                    min_ms: r.min_ms,
                    mean_ms: r.mean_ms,
                    max_ms: r.max_ms,
                    share: r.share,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perf_on() -> Perf {
        let mut p = Perf::new(true, true);
        p.window_s = 0.0; // печатать сразу по требованию
        p.skip = 0; // в тестах первый такт не греем
        p
    }

    #[test]
    fn first_tick_after_enable_is_dropped() {
        let mut p = Perf::new(true, true);
        p.window_s = 0.0;
        // Первый такт — компиляция пайплайнов (сотни мс), в статистику не идёт.
        p.record(&[0], &[500.0], 500.0);
        p.record(&[0], &[2.0], 2.0);
        p.finish_window();
        assert!((p.rows[0].mean_ms - 2.0).abs() < 1e-4);
        assert!((p.rows[0].min_ms - 2.0).abs() < 1e-4);
        assert!((p.rows[0].max_ms - 2.0).abs() < 1e-4);
        assert!((p.tick_mean_ms - 2.0).abs() < 1e-4);
        assert!((p.tick_min_window_ms - 2.0).abs() < 1e-4);
    }

    #[test]
    fn mean_max_and_share_are_per_tick() {
        let mut p = perf_on();
        // Два такта по 10 мс: солвер 6 и 8 мс, интегратор 2 и 2 мс.
        p.record(&[7, 0], &[6.0, 4.0], 10.0);
        p.record(&[7, 0], &[8.0, 2.0], 10.0);
        p.finish_window();
        assert_eq!(p.rows.len(), 2);
        let s = &p.rows[0];
        assert_eq!(s.stage, 7);
        assert!((s.min_ms - 6.0).abs() < 1e-4, "минимум {}", s.min_ms);
        assert!((s.mean_ms - 7.0).abs() < 1e-4, "среднее {}", s.mean_ms);
        assert!((s.max_ms - 8.0).abs() < 1e-4, "максимум {}", s.max_ms);
        assert!((s.share - 70.0).abs() < 1e-3, "доля {}", s.share);
        // по одному диспатчу на такт — перекоса нет
        assert!((s.imbal - 8.0 / 7.0).abs() < 1e-3, "imbal {}", s.imbal);
        assert!((p.tick_min_window_ms - 10.0).abs() < 1e-4);
        assert!((p.tick_mean_ms - 10.0).abs() < 1e-4);
        assert!((p.tick_max_window_ms - 10.0).abs() < 1e-4);
    }

    #[test]
    fn min_is_lowest_tick_of_that_stage_and_tick_range_covers_window() {
        let mut p = perf_on();
        // Такты 10 и 12 мс: стадия 7 — 6 и 9 мс, стадия 3 — 1.5 мс только в
        // первом такте (условный диспатч).
        p.record(&[7, 3], &[6.0, 1.5], 10.0);
        p.record(&[7], &[9.0], 12.0);
        p.finish_window();
        let s7 = p.rows.iter().find(|r| r.stage == 7).unwrap();
        assert!((s7.min_ms - 6.0).abs() < 1e-4, "min {}", s7.min_ms);
        assert!((s7.mean_ms - 7.5).abs() < 1e-4, "mean {}", s7.mean_ms);
        assert!((s7.max_ms - 9.0).abs() < 1e-4, "max {}", s7.max_ms);
        // Пропущенный такт НЕ считается нулём: минимум стадии — по тем тактам,
        // где она правда работала, иначе в таблице был бы мусор «0.000».
        let s3 = p.rows.iter().find(|r| r.stage == 3).unwrap();
        assert!((s3.min_ms - 1.5).abs() < 1e-4, "min {}", s3.min_ms);
        assert!((s3.max_ms - 1.5).abs() < 1e-4, "max {}", s3.max_ms);
        // ИТОГО такта — тоже три числа.
        assert_eq!(p.tick_range_ms().0, 10.0);
        assert!((p.tick_range_ms().1 - 11.0).abs() < 1e-4);
        assert_eq!(p.tick_range_ms().2, 12.0);
        // Окно закрыто — накопители сброшены, второй проход не тянет старое.
        p.finish_window();
        assert!(p.rows.is_empty());
        assert_eq!(p.tick_min_window_ms, 0.0);
    }

    #[test]
    fn multiple_dispatches_of_a_stage_sum_per_tick_and_show_imbalance() {
        let mut p = perf_on();
        // Стадия 0: 12 диспатчей за такт, один тяжёлый (5 мс), остальные по 0.1.
        let mut stages = vec![0u32; 12];
        stages.push(1);
        let mut ms = vec![0.1f32; 12];
        ms[3] = 5.0;
        ms.push(1.0);
        p.record(&stages, &ms, 7.0);
        p.finish_window();
        let s = p.rows.iter().find(|r| r.stage == 0).unwrap();
        assert!((s.mean_ms - (5.0 + 0.1 * 11.0)).abs() < 1e-3);
        assert!((s.max_ms - 6.1).abs() < 1e-3);
        // средний диспатч стадии ~0.508 мс, максимум 5 мс -> перекос ~9.8
        assert!(s.imbal > 9.0 && s.imbal < 11.0, "imbal {}", s.imbal);
    }

    #[test]
    fn disabled_perf_ignores_samples() {
        let mut p = Perf::new(true, false);
        p.record(&[1, 2], &[1.0, 2.0], 4.0);
        assert!(p.rows.is_empty());
        assert_eq!(p.ticks, 0);
    }

    #[test]
    fn unknown_stage_ids_do_not_panic() {
        let mut p = perf_on();
        p.record(&[u32::MAX], &[1.0], 2.0);
        p.finish_window();
        assert!(p.rows.is_empty());
    }

    /// Окно частот: fps считает кадры, tps — завершённые такты, оба с крайними
    /// значениями; среднее время кадра — из тех же кадров.
    #[test]
    fn rates_count_frames_ticks_and_extremes() {
        let t0 = Instant::now();
        let mut r = Rates::new();
        r.skip = 0;
        r.start = t0;
        r.window_s = 0.1;
        // 5 кадров по 20 мс: окно 100 мс закрывается на пятом. Времена кадров
        // 10..14 мс — разброс даёт мин/макс мгновенного fps. Такт — на первом
        // и на последнем кадре: 2 такта на 100 мс.
        let mut closed = false;
        for i in 0..5 {
            let ticks = u32::from(i == 0 || i == 4);
            closed |= r.sample(t0 + Duration::from_millis(20 * (i + 1)), 10.0 + i as f32, ticks);
        }
        assert!(closed, "окно должно было закрыться");
        let o = r.out();
        assert_eq!(o.frames, 5);
        assert_eq!(o.ticks, 2);
        assert!((o.fps_mean - 50.0).abs() < 1e-3, "fps_mean {}", o.fps_mean);
        assert!((o.fps_min - 1000.0 / 14.0).abs() < 1e-2, "fps_min {}", o.fps_min);
        assert!((o.fps_max - 100.0).abs() < 1e-2, "fps_max {}", o.fps_max);
        assert!((o.tps_mean - 20.0).abs() < 1e-2, "tps_mean {}", o.tps_mean);
        // TPS по кадрам: пока такт один — 1, после второго — 2.
        assert!((o.tps_min - 1.0).abs() < 1e-4, "tps_min {}", o.tps_min);
        assert!((o.tps_max - 2.0).abs() < 1e-4, "tps_max {}", o.tps_max);
        assert!((o.frame_mean_ms - 12.0).abs() < 1e-3, "frame {}", o.frame_mean_ms);
    }

    /// Мёртвая физика (ни одного такта) — tps 0, а не пустая строка.
    #[test]
    fn rates_without_ticks_report_zero_tps() {
        let t0 = Instant::now();
        let mut r = Rates::new();
        r.skip = 0;
        r.start = t0;
        r.window_s = 0.05;
        for i in 0..4 {
            r.sample(t0 + Duration::from_millis(20 * (i + 1)), 20.0, 0);
        }
        let o = r.out();
        assert_eq!(o.ticks, 0);
        assert_eq!(o.tps_mean, 0.0);
        assert!((o.fps_mean - 50.0).abs() < 1e-3);
    }

    /// История для графиков: старые отсчёты первыми, длиннее HIST_MAX не растёт,
    /// в конце — самые свежие кадры.
    #[test]
    fn history_keeps_last_hundred_twenty_frames() {
        let t0 = Instant::now();
        let mut r = Rates::new();
        r.skip = 0;
        r.start = t0;
        r.window_s = 10.0; // окно в тесте не закрываем
        for i in 0..(HIST_MAX + 10) {
            r.sample(t0 + Duration::from_millis(i as u64 + 1), 10.0 + i as f32, 0);
        }
        let v = r.c_view();
        assert_eq!(v.fps_hist_n as usize, HIST_MAX);
        let hist = unsafe { std::slice::from_raw_parts(v.fps_hist, v.fps_hist_n as usize) };
        /* Кадры в тесте становятся медленнее (10, 11, … мс), поэтому fps
         * падает: первый отсчёт в окне — самый быстрый, последний — самый
         * медленный, и последний из них — последний поданный кадр. */
        assert!((hist[0] - 1000.0 / 20.0).abs() < 1e-3, "начало {}", hist[0]);
        let want = 1000.0 / (10.0 + (HIST_MAX + 9) as f32);
        assert!((hist[HIST_MAX - 1] - want).abs() < 1e-3, "хвост {}", hist[HIST_MAX - 1]);
        assert!(hist[0] > hist[HIST_MAX - 1]);
    }

    /// Первые кадры (окно, компиляция пайплайнов) в статистику не идут.
    #[test]
    fn rates_skip_warmup_frames() {
        let t0 = Instant::now();
        let mut r = Rates::new();
        r.start = t0;
        r.window_s = 1.0;
        // skip по умолчанию 3: три кадра по 300 мс — это компиляция, не рендер.
        for i in 0..3 {
            assert!(!r.sample(t0 + Duration::from_millis(300 * (i + 1)), 300.0, 1));
        }
        assert_eq!(r.out().frames, 0);
        r.sample(t0 + Duration::from_millis(1300), 16.0, 1);
        assert_eq!(r.frames, 1);
        assert!((r.frame_max_ms - 16.0).abs() < 1e-4);
    }

    /// Один кадр в окне: окно длиннее кадра, поэтому крайние значения взяты
    /// средним — иначе в строке было бы «fps 0.2 (min 0.4 max 0.4)».
    #[test]
    fn single_frame_window_uses_mean_for_extremes() {
        let t0 = Instant::now();
        let mut r = Rates::new();
        r.skip = 0;
        r.start = t0;
        r.window_s = 0.1;
        // Один кадр 400 мс, окно закрывается сразу после него.
        assert!(r.sample(t0 + Duration::from_millis(400), 400.0, 1));
        let o = r.out();
        assert_eq!(o.frames, 1);
        assert!((o.fps_mean - 2.5).abs() < 1e-3, "fps_mean {}", o.fps_mean);
        assert!((o.fps_min - o.fps_mean).abs() < 1e-4);
        assert!((o.fps_max - o.fps_mean).abs() < 1e-4);
        assert!((o.tps_min - o.tps_mean).abs() < 1e-4);
        assert!((o.tps_max - o.tps_mean).abs() < 1e-4);
    }
}
