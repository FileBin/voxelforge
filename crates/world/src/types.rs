//! Базовые типы мира: форма, тело, ГПСЧ, запись траекторий.
//!
//! Это часть движка, а не тестов: CPU-эталон (`tests/refsim`) их импортирует,
//! чтобы сравнивать с GPU теми же числами.

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
    /// Ориентация как кватернион `[w, x, y, z]`. У сферы визуально не видна,
    /// но без неё нет угловой динамики.
    pub rot: [f64; 4],
    /// Угловая скорость, рад/с.
    pub ang: [f64; 3],
    /// Обратный момент инерции, изотропный скаляр. `0` — вращение выключено
    /// (блоки: их опрокидывание требует контактных манифолдов, отдельный шаг).
    pub inv_inertia: f64,
    /// Статическое тело (бесконечная масса) — пол, планета.
    pub anchored: bool,
}

impl Body {
    pub fn new(pos: [f64; 3], vel: [f64; 3], mass: f64, radius: f32) -> Self {
        let r = radius as f64;
        // Момент инерции однородной сферы 2/5·m·r².
        let inv_inertia = if mass > 0.0 && r > 0.0 {
            1.0 / (0.4 * mass * r * r)
        } else {
            0.0
        };
        Self {
            pos,
            vel,
            mass,
            radius,
            half: [radius as f64; 3],
            shape: Shape::Sphere,
            rot: [1.0, 0.0, 0.0, 0.0],
            ang: [0.0; 3],
            inv_inertia,
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
            rot: [1.0, 0.0, 0.0, 0.0],
            ang: [0.0; 3],
            // Вращения блоков пока нет: оно требует контактных манифолдов
            // (несколько точек на грань), а не одной точки.
            inv_inertia: 0.0,
            anchored: false,
        }
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
    use super::{Body, Shape};

    pub const MAGIC: &[u8; 4] = b"VFR2";

    pub fn encode(bodies: &[Body], frames: &[Vec<[f32; 3]>]) -> Vec<u8> {
        let n = bodies.len();
        let mut out = Vec::with_capacity(12 + n * 21 + frames.len() * n * 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        for b in bodies {
            out.extend_from_slice(&b.mass.to_le_bytes());
            out.extend_from_slice(&b.radius.to_le_bytes());
            out.push(match b.shape {
                Shape::Sphere => 0u8,
                Shape::Block => 1u8,
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

