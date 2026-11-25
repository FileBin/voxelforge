//! CPU-эталон физики: гравитация, интегратор, контакты, запись траекторий.
//!
//! Это **оракул**, а не продакшн-путь. Он существует, чтобы:
//! 1. давать числа для приёмочных тестов, пока GPU-версии нет;
//! 2. служить эталоном сравнения, когда она появится (одни и те же сценарии);
//! 3. позволять смотреть симуляцию без GPU и без окна — через записанный трек.
//!
//! Всё в `f64`, суммирование в фиксированном порядке индексов: два прогона с
//! одинаковым входом обязаны дать побайтово одинаковую запись (тесты
//! `deterministic_recording` и `contact_sim_deterministic`). Трансцендентные
//! функции (`sqrt`) здесь допустимы — эталон живёт на одной машине и ни с кем
//! не синхронизируется; в GPU-солвере они, наоборот, запрещены гейтами.

pub mod contact;
pub mod pile;

pub use contact::ContactSim;
pub use pile::{pile_demo, PileSim};

/// Форма тела: сфера (гравитационные демки) или блок (контактные демки).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Sphere,
    Block,
}

/// Тело: масса, положение, форма. Вращения пока нет (см. модуль `contact`).
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub radius: f32,
    /// Половина размеров AABB: для сферы — `[r, r, r]`.
    pub half: [f64; 3],
    pub shape: Shape,
    /// Статическое тело (бесконечная масса) — пол, планета.
    pub anchored: bool,
}

impl Body {
    pub fn new(pos: [f64; 3], vel: [f64; 3], mass: f64, radius: f32) -> Self {
        Self {
            pos,
            vel,
            mass,
            radius,
            half: [radius as f64; 3],
            shape: Shape::Sphere,
            anchored: false,
        }
    }

    /// Коробка по половине размеров; масса — из плотности 1.
    pub fn block(pos: [f64; 3], half: [f64; 3]) -> Self {
        let volume = 8.0 * half[0] * half[1] * half[2];
        let radius = half[0].max(half[1]).max(half[2]) as f32;
        Self {
            pos,
            vel: [0.0; 3],
            mass: volume,
            radius,
            half,
            shape: Shape::Block,
            anchored: false,
        }
    }
}

/// Симуляция: N тел, ньютоновская гравитация, leapfrog (kick-drift-kick).
pub struct Sim {
    pub bodies: Vec<Body>,
    pub g: f64,
    pub softening: f64,
}

impl Sim {
    pub fn new(bodies: Vec<Body>) -> Self {
        Self { bodies, g: 1.0, softening: 0.0 }
    }

    /// Ускорения всех тел. O(N²), фиксированный порядок суммирования.
    fn accel(&self) -> Vec<[f64; 3]> {
        let n = self.bodies.len();
        let mut out = vec![[0.0f64; 3]; n];
        for i in 0..n {
            let bi = &self.bodies[i];
            let mut a = [0.0f64; 3];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let bj = &self.bodies[j];
                let dx = bj.pos[0] - bi.pos[0];
                let dy = bj.pos[1] - bi.pos[1];
                let dz = bj.pos[2] - bi.pos[2];
                let r2 = dx * dx + dy * dy + dz * dz + self.softening * self.softening;
                if r2 == 0.0 {
                    continue;
                }
                let inv_r = 1.0 / r2.sqrt();
                let s = self.g * bj.mass * inv_r * inv_r * inv_r;
                a[0] += s * dx;
                a[1] += s * dy;
                a[2] += s * dz;
            }
            out[i] = a;
        }
        out
    }

    /// Один шаг leapfrog KDK: kick(½) → drift → kick(½).
    pub fn step(&mut self, dt: f64) {
        let a0 = self.accel();
        for (b, a) in self.bodies.iter_mut().zip(a0.iter()) {
            if b.anchored {
                continue;
            }
            for k in 0..3 {
                b.vel[k] += 0.5 * dt * a[k];
                b.pos[k] += dt * b.vel[k];
            }
        }
        let a1 = self.accel();
        for (b, a) in self.bodies.iter_mut().zip(a1.iter()) {
            if b.anchored {
                continue;
            }
            for k in 0..3 {
                b.vel[k] += 0.5 * dt * a[k];
            }
        }
    }

    pub fn kinetic(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| !b.anchored)
            .map(|b| {
                let v2 = b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2];
                0.5 * b.mass * v2
            })
            .sum()
    }

    pub fn potential(&self) -> f64 {
        let n = self.bodies.len();
        let eps2 = self.softening * self.softening;
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let bi = &self.bodies[i];
                let bj = &self.bodies[j];
                if bi.anchored && bj.anchored {
                    continue;
                }
                let dx = bj.pos[0] - bi.pos[0];
                let dy = bj.pos[1] - bi.pos[1];
                let dz = bj.pos[2] - bi.pos[2];
                let r = (dx * dx + dy * dy + dz * dz + eps2).sqrt();
                u -= self.g * bi.mass * bj.mass / r;
            }
        }
        u
    }

    pub fn energy(&self) -> f64 {
        self.kinetic() + self.potential()
    }

    pub fn momentum(&self) -> [f64; 3] {
        let mut p = [0.0f64; 3];
        for b in self.bodies.iter().filter(|b| !b.anchored) {
            for k in 0..3 {
                p[k] += b.mass * b.vel[k];
            }
        }
        p
    }
}

/// Детерминированный ГПСЧ (LCG), чтобы сценарии воспроизводились.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(2) | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Равномерно в [-1, 1).
    pub fn signed(&mut self) -> f64 {
        (self.next_u32() as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// Формат записи траектории `.vfr`, версия 2 (с формой тела):
///
/// `"VFR2" | u32 bodies | u32 frames | [f64 mass, f32 radius, u8 shape, f32 half×3]×n | f32 pos×3 × frames`
pub mod vfr {
    pub const MAGIC: &[u8; 4] = b"VFR2";

    pub fn encode(bodies: &[crate::Body], frames: &[Vec<[f32; 3]>]) -> Vec<u8> {
        let n = bodies.len();
        let mut out = Vec::with_capacity(12 + n * 21 + frames.len() * n * 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        for b in bodies {
            out.extend_from_slice(&b.mass.to_le_bytes());
            out.extend_from_slice(&b.radius.to_le_bytes());
            out.push(match b.shape {
                crate::Shape::Sphere => 0u8,
                crate::Shape::Block => 1u8,
            });
            for k in 0..3 {
                out.extend_from_slice(&(b.half[k] as f32).to_le_bytes());
            }
        }
        for f in frames {
            for p in f {
                out.extend_from_slice(&p[0].to_le_bytes());
                out.extend_from_slice(&p[1].to_le_bytes());
                out.extend_from_slice(&p[2].to_le_bytes());
            }
        }
        out
    }
}

/// Простенький SVG-снимок: траектории и текущие положения (вид сбоку, x–y).
pub fn svg(bodies: &[Body], frames: &[Vec<[f32; 3]>], size: u32) -> String {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for f in frames {
        for p in f {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    // Учитываем размеры тел, иначе крайние блоки обрезаются рамкой.
    let pad = bodies
        .iter()
        .map(|b| (b.half[0] as f32).max(b.half[1] as f32))
        .fold(0.0f32, f32::max);
    let cx = 0.5 * (lo[0] + hi[0]);
    let cy = 0.5 * (lo[1] + hi[1]);
    let span = ((hi[0] - lo[0]).max(hi[1] - lo[1]) + 2.0 * pad).max(1.0) * 0.55;
    let s = (size as f32 * 0.5) / span;
    let px = |x: f32| size as f32 * 0.5 + (x - cx) * s;
    let py = |y: f32| size as f32 * 0.5 - (y - cy) * s;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{0}" viewBox="0 0 {0} {0}"><rect width="{0}" height="{0}" fill="#0b0f14"/>"##,
        size
    ));
    let ground_y = py(0.0);
    if ground_y >= 0.0 && ground_y <= size as f32 {
        out.push_str(&format!(
            r##"<line x1="0" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#1e2b38" stroke-width="1"/>"##,
            ground_y, size, ground_y
        ));
    }
    let palette = ["#59a5ff", "#ffb347", "#7ee08a", "#ff6b8a", "#c48aff", "#4fd6d6"];
    // Траектории рисуем только для малого числа тел: иначе SVG раздувается.
    if bodies.len() <= 64 {
        for (i, _b) in bodies.iter().enumerate() {
            let color = palette[i % palette.len()];
            let mut d = String::new();
            for (fi, f) in frames.iter().enumerate() {
                if fi % 2 != 0 {
                    continue;
                }
                let p = f[i];
                let cmd = if d.is_empty() { 'M' } else { 'L' };
                d.push_str(&format!("{}{:.1},{:.1} ", cmd, px(p[0]), py(p[1])));
            }
            out.push_str(&format!(
                r#"<path d="{}" fill="none" stroke="{}" stroke-opacity="0.55" stroke-width="1"/>"#,
                d, color
            ));
        }
    }
    if let Some(last) = frames.last() {
        for (i, b) in bodies.iter().enumerate() {
            let p = last[i];
            let color = palette[i % palette.len()];
            match b.shape {
                Shape::Block => {
                    let hw = (b.half[0] as f32 * s).max(1.0);
                    let hh = (b.half[1] as f32 * s).max(1.0);
                    out.push_str(&format!(
                        r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#,
                        px(p[0]) - hw,
                        py(p[1]) - hh,
                        2.0 * hw,
                        2.0 * hh,
                        color
                    ));
                }
                Shape::Sphere => {
                    let r = (b.radius * s).clamp(1.5, 14.0);
                    out.push_str(&format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}"/>"#,
                        px(p[0]),
                        py(p[1]),
                        r,
                        color
                    ));
                }
            }
        }
    }
    out.push_str("</svg>");
    out
}

