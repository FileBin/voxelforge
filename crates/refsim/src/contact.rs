//! Контактный солвер эталона: AABB-тела, пол-плоскость, XPBD + подшаги.
//!
//! Что здесь проверяется и что переносится на GPU:
//! * контакты считаются в одном проходе и раскладываются по `axis` — тот же
//!   gather без атомиков, что будет в шейдере;
//! * подшаги вместо итераций: фиксированное число операций на поток, поэтому
//!   работа равномерная (главная причина, почему обломки плохо ложатся на GPU);
//! * жёсткий детерминизм: никаких трансцендентных функций, фиксированный
//!   порядок обхода пар, скорости выводятся из позиций.
//!
//! Чего ещё нет и что будет следующим инкрементом: **вращения тел** (поэтому
//! коробки не могут опрокинуться и всегда остаются осевыми), трения, сна,
//! сетки broadphase (здесь O(N²), что для эталона на сотни тел достаточно).
//! Отсутствие вращения — осознанное упрощение: сначала нужно, чтобы сходились
//! нормальные контакты, иначе отладка смешивает две ошибки.

use crate::Body;

const GROUND: usize = usize::MAX;

/// Контакт: пара тел и ось разделения. `a == GROUND` — контакт с полом.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub a: usize,
    pub b: usize,
    pub axis: usize,
}

/// Симуляция с контактами: постоянная гравитация вниз, пол-плоскость `ground_y`.
pub struct ContactSim {
    pub bodies: Vec<Body>,
    pub gravity: f64,
    pub ground_y: f64,
    pub substeps: usize,
    pub iterations: usize,
    /// Контактный зазор: проникновение меньше этого не корректируется.
    ///
    /// Без него тело, выставленное ровно в касание, каждый подшаг снова падает
    /// на `g·h²` и его тут же выталкивает обратно — позиция дрожит на десятые
    /// доли миллиметра, а скорость, выведенная как `Δx/h`, при `h ≈ 2 мс`
    /// превращает эту дрожь в «0.1 м/с покоя». Это ровно то, что делают
    /// реальные движки (contact slop): Box2D берёт 5 мм, здесь 0.5 мм.
    pub margin: f64,
    pub contacts: Vec<Contact>,
}

impl ContactSim {
    pub fn new(bodies: Vec<Body>) -> Self {
        Self {
            bodies,
            gravity: 9.81,
            ground_y: 0.0,
            substeps: 8,
            iterations: 6,
            margin: 5e-4,
            contacts: Vec::new(),
        }
    }

    fn inv_mass(&self, i: usize) -> f64 {
        if i == GROUND || self.bodies[i].anchored || self.bodies[i].mass <= 0.0 {
            0.0
        } else {
            1.0 / self.bodies[i].mass
        }
    }

    /// Глубина проникновения по зафиксированной оси контакта.
    fn penetration(&self, c: Contact) -> f64 {
        if c.a == GROUND {
            let b = &self.bodies[c.b];
            return self.ground_y - (b.pos[1] - b.half[1]);
        }
        let a = &self.bodies[c.a];
        let b = &self.bodies[c.b];
        (a.half[c.axis] + b.half[c.axis]) - (b.pos[c.axis] - a.pos[c.axis]).abs()
    }

    /// Собирает контакты. Порядок обхода фиксирован (пол, затем пары i<j).
    fn collect_contacts(&mut self) {
        self.contacts.clear();
        let n = self.bodies.len();

        for i in 0..n {
            let b = &self.bodies[i];
            if b.anchored {
                continue;
            }
            if b.pos[1] - b.half[1] < self.ground_y {
                self.contacts.push(Contact { a: GROUND, b: i, axis: 1 });
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&self.bodies[i], &self.bodies[j]);
                if a.anchored && b.anchored {
                    continue;
                }
                let mut min_overlap = f64::INFINITY;
                let mut axis = 0usize;
                let mut touching = true;
                for k in 0..3 {
                    let overlap = (a.half[k] + b.half[k]) - (b.pos[k] - a.pos[k]).abs();
                    if overlap <= 0.0 {
                        touching = false;
                        break;
                    }
                    if overlap < min_overlap {
                        min_overlap = overlap;
                        axis = k;
                    }
                }
                if touching {
                    self.contacts.push(Contact { a: i, b: j, axis });
                }
            }
        }
    }

    /// Одна проекция контакта: XPBD с нулевой податливостью, Якоби-стиль
    /// (каждый контакт пишет только в свои тела).
    fn project(&mut self, c: Contact) {
        // Контактный зазор: корректируем только то, что глубже margin.
        let pen = self.penetration(c) - self.margin;
        if pen <= 0.0 {
            return;
        }
        let (wa, wb) = (self.inv_mass(c.a), self.inv_mass(c.b));
        let w = wa + wb;
        if w <= 0.0 {
            return;
        }
        let k = c.axis;

        // Направление, в котором тела расходятся.
        let dir = if c.a == GROUND {
            1.0
        } else if self.bodies[c.b].pos[k] >= self.bodies[c.a].pos[k] {
            1.0
        } else {
            -1.0
        };

        let step = pen / w * dir;
        if wa > 0.0 {
            self.bodies[c.a].pos[k] -= wa * step;
        }
        if wb > 0.0 {
            self.bodies[c.b].pos[k] += wb * step;
        }
    }

    fn substep(&mut self, h: f64) {
        let prev: Vec<[f64; 3]> = self.bodies.iter().map(|b| b.pos).collect();

        for b in self.bodies.iter_mut() {
            if b.anchored {
                continue;
            }
            b.vel[1] -= self.gravity * h;
            for k in 0..3 {
                b.pos[k] += b.vel[k] * h;
            }
        }

        self.collect_contacts();
        for _ in 0..self.iterations {
            for ci in 0..self.contacts.len() {
                let c = self.contacts[ci];
                self.project(c);
            }
        }

        // Скорость выводится из фактического перемещения: это и даёт
        // неупругий удар (e = 0) без отдельной формулы.
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if b.anchored {
                continue;
            }
            for k in 0..3 {
                b.vel[k] = (b.pos[k] - prev[i][k]) / h;
            }
        }
    }

    /// Шаг симуляции: `dt` делится на `substeps` подшагов.
    pub fn step(&mut self, dt: f64) {
        let h = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            self.substep(h);
        }
    }

    /// Самая глубокая просадка в текущем состоянии (>0 — тела проникли).
    pub fn max_penetration(&mut self) -> f64 {
        self.collect_contacts();
        let mut worst: f64 = 0.0;
        for ci in 0..self.contacts.len() {
            let pen = self.penetration(self.contacts[ci]);
            if pen > worst {
                worst = pen;
            }
        }
        worst
    }

    /// Максимальная скорость среди тел.
    pub fn max_speed(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| !b.anchored)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max)
    }

    /// Самая высокая точка мира (верх самого верхнего тела).
    pub fn top(&self) -> f64 {
        self.bodies
            .iter()
            .map(|b| b.pos[1] + b.half[1])
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drop_scene() -> ContactSim {
        let mut sim = ContactSim::new(vec![Body::block([0.0, 3.0, 0.0], [0.5, 0.5, 0.5])]);
        sim.gravity = 9.81;
        sim.substeps = 8;
        sim.iterations = 6;
        sim
    }

    fn stack_scene(count: usize) -> ContactSim {
        let half = [0.5, 0.2, 0.5];
        let mut bodies = Vec::new();
        for i in 0..count {
            let y = half[1] + i as f64 * (2.0 * half[1] + 0.002);
            bodies.push(Body::block([0.0, y, 0.0], half));
        }
        let mut sim = ContactSim::new(bodies);
        sim.gravity = 9.81;
        sim.substeps = 8;
        sim.iterations = 8;
        sim
    }

    /// D1: коробка падает на пол и остаётся стоять — без просадки и дрожания.
    #[test]
    fn box_lands_and_rests() {
        let mut sim = drop_scene();
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            sim.step(dt);
        }
        let rest_y = sim.bodies[0].pos[1];
        let pen = sim.max_penetration();
        let speed = sim.max_speed();
        assert!((rest_y - 0.5).abs() < 0.02, "rest height {rest_y:.4}, ожидалось ~0.5");
        assert!(pen < 0.01, "просадка {:.4} м > 1 см", pen);
        assert!(speed < 0.01, "скорость покоя {speed:.4} м/с > 1 см/с");
    }

    /// D2: стопка из 10 блоков стоит 10 секунд и не складывается.
    #[test]
    fn stack_of_ten_stands() {
        let mut sim = stack_scene(10);
        let top0 = sim.top();
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            sim.step(dt);
        }
        let top1 = sim.top();
        let sag = (top0 - top1) / top0;
        let pen = sim.max_penetration();
        let speed = sim.max_speed();
        assert!(sag < 0.05, "просадка стопки {:.2}% > 5%", 100.0 * sag);
        assert!(pen < 0.02, "проникновение в стопке {:.4} м > 2 см", pen);
        assert!(speed < 0.02, "стопка не успокоилась: {speed:.4} м/с");
    }

    /// Контакты не должны менять импульс системы по горизонтали.
    #[test]
    fn contact_sim_is_deterministic() {
        let run = || {
            let mut sim = stack_scene(6);
            let dt = 1.0 / 60.0;
            for _ in 0..90 {
                sim.step(dt);
            }
            sim.bodies
                .iter()
                .flat_map(|b| b.pos.iter().map(|v| v.to_bits()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(run(), run(), "два прогона дали разные позиции");
    }
}
