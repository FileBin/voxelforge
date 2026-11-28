//! Диагностика производительности: агрегация GPU-таймстемпов по стадиям такта.
//!
//! Ядро (`c/voxelforge_core.c`) пишет по метке до каждого диспатча и после
//! последнего, а результаты отдаёт на СЛЕДУЮЩЕМ такте — внутри такта ничего не
//! ждётся. Здесь из этих длительностей собирается окно в одну секунду:
//! среднее, максимум и доля от такта на каждую стадию, плюс неровномерность
//! диспетчей внутри стадии (макс/среднее) — она показывает перекос нагрузки,
//! который средние скрывают.
//!
//! Печать — одна таблица раз в секунду, самой дорогой стадией сверху: вопрос
//! «что оптимизировать» должен решаться чтением первой строки.

use std::time::{Duration, Instant};

use crate::sim::PIPE_NAMES;

/// Сколько стадий вообще может прийти: пайплайнов сейчас `PIPE_NAMES.len()`.
const MAX_STAGES: usize = 64;

#[derive(Clone, Copy, Default)]
struct StageAcc {
    /// Сумма времени стадии за такт — по всем тактам окна.
    tick_sum_ms: f64,
    /// Максимум времени стадии за один такт.
    tick_max_ms: f32,
    /// Сумма длительностей отдельных диспатчей (для среднего по диспетчам).
    disp_sum_ms: f64,
    /// Самый долгий отдельный диспатч стадии.
    disp_max_ms: f32,
    /// Сколько раз стадия встречалась в окне (диспатчей).
    disp_n: u32,
}

/// Одна строка таблицы — то, что печатается и отдаётся оверлею.
#[derive(Clone, Copy)]
pub struct PerfRow {
    pub stage: usize,
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
    pub mean_ms: f32,
    pub max_ms: f32,
    pub share: f32,
}

pub struct Perf {
    /// Драйвер умеет таймстемпы на этой очереди.
    pub supported: bool,
    /// Сбор включён (F3 / VF_PERF=1).
    pub enabled: bool,
    acc: [StageAcc; MAX_STAGES],
    /// Сколько тактов попало в окно.
    ticks: u32,
    tick_sum_ms: f64,
    tick_max_ms: f32,
    /// С какого момента копится окно.
    window_start: Instant,
    /// Готовые строки последнего окна — для оверлея в окне.
    pub rows: Vec<PerfRow>,
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
            acc: [StageAcc::default(); MAX_STAGES],
            ticks: 0,
            tick_sum_ms: 0.0,
            tick_max_ms: 0.0,
            window_start: Instant::now(),
            rows: Vec::new(),
            tick_mean_ms: 0.0,
            tick_max_window_ms: 0.0,
            window_s: 1.0,
            skip: 1,
        }
    }

    /// Один закрытый такт: длительности диспатчей и их номера стадий.
    /// `tick_ms` — от первой метки до последней.
    pub fn record(&mut self, stages: &[u32], ms: &[f32], tick_ms: f32) {
        if !self.enabled || stages.is_empty() {
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
        self.tick_sum_ms += tick_ms.max(0.0) as f64;
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
                    mean_ms: mean,
                    max_ms: a.tick_max_ms,
                    share: if tick_mean > 0.0 { 100.0 * mean / tick_mean } else { 0.0 },
                    imbal: if disp_mean > 1e-9 { a.disp_max_ms / disp_mean } else { 1.0 },
                }
            })
            .collect();
        rows.sort_by(|a, b| b.mean_ms.total_cmp(&a.mean_ms));
        self.rows = rows;
        self.tick_mean_ms = tick_mean;
        self.tick_max_window_ms = self.tick_max_ms;
        self.acc = [StageAcc::default(); MAX_STAGES];
        self.ticks = 0;
        self.tick_sum_ms = 0.0;
        self.tick_max_ms = 0.0;
    }

    fn print_table(&self) {
        println!(
            "[perf] такт: среднее {:.2} мс · макс {:.2} мс · стадии по убыванию",
            self.tick_mean_ms, self.tick_max_window_ms
        );
        println!(
            "  {:<18} {:>10} {:>9} {:>7} {:>7}",
            "стадия", "среднее", "макс", "доля%", "дисп"
        );
        for r in &self.rows {
            let name = PIPE_NAMES.get(r.stage).copied().unwrap_or("?");
            println!(
                "  {:<18} {:>10.3} {:>9.3} {:>6.1}% {:>7.2}",
                name, r.mean_ms, r.max_ms, r.share, r.imbal
            );
        }
        println!("  {:<18} {:>10.3} {:>9.3} {:>6.1}%", "ИТОГО такт", self.tick_mean_ms, self.tick_max_window_ms, 100.0);
    }

    /// Начать окно заново (включение сбора с чистого листа).
    pub fn restart_window(&mut self) {
        self.window_start = Instant::now();
        self.acc = [StageAcc::default(); MAX_STAGES];
        self.ticks = 0;
        self.tick_sum_ms = 0.0;
        self.tick_max_ms = 0.0;
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
        assert!((p.rows[0].max_ms - 2.0).abs() < 1e-4);
        assert!((p.tick_mean_ms - 2.0).abs() < 1e-4);
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
        assert!((s.mean_ms - 7.0).abs() < 1e-4, "среднее {}", s.mean_ms);
        assert!((s.max_ms - 8.0).abs() < 1e-4, "максимум {}", s.max_ms);
        assert!((s.share - 70.0).abs() < 1e-3, "доля {}", s.share);
        // по одному диспатчу на такт — перекоса нет
        assert!((s.imbal - 8.0 / 7.0).abs() < 1e-3, "imbal {}", s.imbal);
        assert!((p.tick_mean_ms - 10.0).abs() < 1e-4);
        assert!((p.tick_max_window_ms - 10.0).abs() < 1e-4);
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
}
