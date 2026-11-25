//! CPU-эталон физики: гравитация, интегратор, запись траекторий.
//!
//! Это **оракул**, а не продакшн-путь. Он существует, чтобы:
//! 1. давать числа для приёмочных тестов, пока GPU-версии нет;
//! 2. служить эталоном сравнения, когда она появится (одни и те же сценарии);
//! 3. позволять смотреть симуляцию без GPU и без окна — через записанный трек.
//!
//! Всё в `f64`, суммирование в фиксированном порядке индексов: два прогона с
//! одинаковым входом обязаны дать побайтово одинаковую запись (это тест
//! `deterministic_recording`). Трансцендентные функции (`sqrt`) здесь
//! допустимы — эталон живёт на одной машине и ни с кем не синхронизируется;
//! в GPU-солвере они, наоборот, запрещены гейтами детерминизма.

/// Тело: точечная масса со сферой визуального радиуса.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub radius: f32,
    /// Статическое тело (бесконечная масса) — для будущих планет.
    pub anchored: bool,
}

impl Body {
    pub fn new(pos: [f64; 3], vel: [f64; 3], mass: f64, radius: f32) -> Self {
        Self { pos, vel, mass, radius, anchored: false }
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

/// Формат записи траектории `.vfr`:
/// magic "VFR1" | u32 bodies | u32 frames | f64 mass[n] | f32 radius[n] | f32 pos[3n] × frames
pub mod vfr {
    pub const MAGIC: &[u8; 4] = b"VFR1";

    pub fn encode(bodies: &[crate::Body], frames: &[Vec<[f32; 3]>]) -> Vec<u8> {
        let n = bodies.len();
        let mut out = Vec::with_capacity(12 + n * 12 + frames.len() * n * 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        for b in bodies {
            out.extend_from_slice(&b.mass.to_le_bytes());
            out.extend_from_slice(&b.radius.to_le_bytes());
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

/// Простенький SVG-снимок: траектории и текущие положения (вид сверху).
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
    let cx = 0.5 * (lo[0] + hi[0]);
    let cy = 0.5 * (lo[1] + hi[1]);
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0) * 0.55;
    let s = (size as f32 * 0.5) / span;
    let px = |x: f32| size as f32 * 0.5 + (x - cx) * s;
    let py = |y: f32| size as f32 * 0.5 - (y - cy) * s;

    let mut out = String::new();
    out.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{0}" viewBox="0 0 {0} {0}"><rect width="{0}" height="{0}" fill="#0b0f14"/>"##,
        size
    ));
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
            let r = (b.radius * s).clamp(1.5, 14.0);
            out.push_str(&format!(
                r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}"/>"#,
                px(p[0]),
                py(p[1]),
                r,
                palette[i % palette.len()]
            ));
        }
    }
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular_orbit() -> (Sim, f64) {
        // Центральное тело массы M и спутник на круговой орбите радиуса r.
        let m: f64 = 1000.0;
        let r: f64 = 10.0;
        let g: f64 = 1.0;
        let v: f64 = (g * m / r).sqrt();
        let star = Body::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], m, 1.0);
        let planet = Body::new([r, 0.0, 0.0], [0.0, 0.0, v], 1e-6, 0.2);
        let mut sim = Sim::new(vec![star, planet]);
        sim.g = g;
        (sim, 2.0 * std::f64::consts::PI * r / v) // период
    }

    #[test]
    fn orbit_closes_after_ten_periods() {
        let (mut sim, period) = circular_orbit();
        let dt = period / 4096.0;
        let start = sim.bodies[1].pos;
        for _ in 0..(4096 * 10) {
            sim.step(dt);
        }
        let end = sim.bodies[1].pos;
        let err = ((end[0] - start[0]).powi(2)
            + (end[1] - start[1]).powi(2)
            + (end[2] - start[2]).powi(2))
        .sqrt();
        let r = (start[0].powi(2) + start[1].powi(2) + start[2].powi(2)).sqrt();
        assert!(err / r < 0.01, "closure error {:.4}%", 100.0 * err / r);
    }

    #[test]
    fn energy_drift_bounded() {
        let (mut sim, period) = circular_orbit();
        sim.softening = 1e-6;
        let dt = period / 4096.0;
        let e0 = sim.energy();
        for _ in 0..(4096 * 10) {
            sim.step(dt);
        }
        let e1 = sim.energy();
        let drift = ((e1 - e0) / e0).abs();
        assert!(drift < 0.01, "energy drift {:.4}%", 100.0 * drift);
    }

    #[test]
    fn momentum_conserved() {
        let mut sim = Sim::new(vec![
            Body::new([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 1.0, 0.2),
            Body::new([-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], 3.0, 0.3),
            Body::new([0.0, 2.0, 0.0], [0.3, 0.0, 0.0], 2.0, 0.25),
        ]);
        let p0 = sim.momentum();
        for _ in 0..2000 {
            sim.step(1e-3);
        }
        let p1 = sim.momentum();
        for k in 0..3 {
            assert!((p1[k] - p0[k]).abs() < 1e-9, "momentum drift on axis {k}");
        }
    }

    #[test]
    fn deterministic_recording() {
        let run = || {
            let mut sim = Sim::new(vec![
                Body::new([1.0, 0.0, 0.0], [0.0, 0.7, 0.0], 5.0, 0.2),
                Body::new([-2.0, 0.5, 0.0], [0.0, -0.3, 0.1], 4.0, 0.2),
            ]);
            let mut frames = Vec::new();
            for _ in 0..64 {
                sim.step(1e-3);
                frames.push(
                    sim.bodies
                        .iter()
                        .map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32])
                        .collect::<Vec<_>>(),
                );
            }
            vfr::encode(&sim.bodies, &frames)
        };
        assert_eq!(run(), run(), "recording must be byte-identical");
    }
}
