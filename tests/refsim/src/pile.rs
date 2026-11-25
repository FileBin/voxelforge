//! Плотная куча: сферы с вращением, скоростной контактный слой, сон, сетка.
//!
//! История решения, чтобы не повторять ошибку. Первой версией был
//! **позиционный** солвер (XPBD-стиль: сдвигаем позиции, скорость выводим как
//! `v = Δx/h`). На разреженных контактах он работает (демки D1/D2), но в
//! плотной куче каждая коррекция проникновения превращается в скорость:
//! коррекции не успевают разрешиться за фиксированные итерации, накапливаются
//! — тела разлетаются до 25 м/с, куча расползается в один слой.
//!
//! Сейчас: **сначала скорости, потом позиции**. Контакты решаются импульсами с
//! bias (Баумгарте), позиции интегрируются из исправленных скоростей. Тогда
//! исправление проникновения — импульс, а не телепорт.
//!
//! Трение считается по скорости **точки контакта** (`v + ω × r`) и в
//! фиксированном касательном базисе контакта. Оба пункта принципиальны:
//! * по точке контакта — иначе шар не начинает катиться, а просто тормозится;
//! * фиксированный базис — если брать направление из мгновенной скорости и
//!   накапливать импульс в нём, при вращении направление «крутится» вместе со
//!   скоростью и солвер раскачивается (проверено: проникновение до 14 см).

use std::collections::HashMap;

use crate::Body;

const GROUND: usize = usize::MAX;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Любая единичная нормаль, перпендикулярная `n`. Выбор детерминированный:
/// от него зависит касательный базис, а значит и воспроизводимость.
fn any_perp(n: [f64; 3]) -> [f64; 3] {
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let c = cross(n, a);
    let len = dot3(c, c).sqrt();
    [c[0] / len, c[1] / len, c[2] / len]
}

/// Фиксированный касательный базис контакта.
fn tangents(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let t1 = any_perp(n);
    (t1, cross(n, t1))
}

/// Интеграция ориентации: `q' = q + ½·(0, ω)⊗q·dt`, затем нормализация.
fn integrate_quat(q: &mut [f64; 4], w: [f64; 3], dt: f64) {
    let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    let hx = w[0] * 0.5 * dt;
    let hy = w[1] * 0.5 * dt;
    let hz = w[2] * 0.5 * dt;
    // (0, h) ⊗ q = (-h·q_v, q_w·h + h×q_v)
    let dw = -(hx * qx + hy * qy + hz * qz);
    let dx = qw * hx + (hy * qz - hz * qy);
    let dy = qw * hy + (hz * qx - hx * qz);
    let dz = qw * hz + (hx * qy - hy * qx);
    let (nw, nx, ny, nz) = (qw + dw, qx + dx, qy + dy, qz + dz);
    let len = (nw * nw + nx * nx + ny * ny + nz * nz).sqrt();
    if len > 0.0 {
        *q = [nw / len, nx / len, ny / len, nz / len];
    }
}

/// Ключ пары для тёплого старта. У статических контактов (пол, четыре стенки)
/// ключ обязан включать **вид** контакта: иначе все они делят один ключ
/// `(GROUND, i)` и импульс стенки применяется к полу — тела у стен
/// разлетаются и застревают друг в друге.
fn contact_key(a: usize, b: usize, kind: u8) -> (u32, u32) {
    if a == GROUND {
        (0xFFFF_0000u32 | kind as u32, b as u32)
    } else if b == GROUND {
        (0xFFFF_0000u32 | kind as u32, a as u32)
    } else {
        (a.min(b) as u32, a.max(b) as u32)
    }
}

/// Контакт сферы с сферой, полом или стенкой.
#[derive(Clone, Copy)]
pub struct PileContact {
    pub a: usize,
    pub b: usize,
    /// Вид контакта: 0 — пара тел, 1 — пол, 2..5 — стенки (−x, +x, −z, +z).
    /// Нужен для ключа тёплого старта.
    pub kind: u8,
    /// Нормаль от `a` к `b`.
    pub n: [f64; 3],
    pub pen: f64,
    /// Фиксированный касательный базис (не зависит от скоростей).
    pub t1: [f64; 3],
    pub t2: [f64; 3],
    /// Накопленные импульсы за подшаг.
    pub lambda_n: f64,
    pub lambda_t1: f64,
    pub lambda_t2: f64,
}

pub struct PileSim {
    pub bodies: Vec<Body>,
    pub gravity: f64,
    pub ground_y: f64,
    pub substeps: usize,
    pub iterations: usize,
    /// Контактный зазор: проникновение мельче него не выталкивается.
    pub margin: f64,
    /// Доля проникновения, выталкиваемая за подшаг (0..1).
    pub beta: f64,
    /// Потолок скорости выталкивания, м/с (max correction velocity).
    pub max_push: f64,
    /// Угловое демпфирование, 1/с — сопротивление качению. Идеальная сфера на
    /// идеальной плоскости катится вечно, и куча никогда не засыпает.
    pub angular_damping: f64,
    /// Коэффициент трения Кулона.
    pub friction: f64,
    /// Стены-«ящик» по x и z: `(min, max)`. Без них шары разъезжаются по
    /// бесконечному полу — они же катятся, — и куча не образуется.
    pub walls: Option<([f64; 3], [f64; 3])>,
    /// Пары с отношением масс выше этого решаются односторонне.
    pub mass_ratio_limit: f64,
    pub sleep_linear: f64,
    pub sleep_time: f64,
    pub cell: f64,
    pub contacts: Vec<PileContact>,
    pub asleep: Vec<bool>,
    sleep_for: Vec<f64>,
    sleep_anchor: Vec<[f64; 3]>,
    pub awake: usize,
    /// Тёплый старт: импульсы контактов с прошлого подшага, по ключу пары.
    /// Без него глубокий штабель не сходится за 8 итераций: нагрузка нижних
    /// слоёв каждый раз считается с нуля, и проникновение копится.
    warm: HashMap<(u32, u32), [f64; 3]>,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl PileSim {
    pub fn new(bodies: Vec<Body>) -> Self {
        let n = bodies.len();
        let cell = bodies
            .iter()
            .map(|b| 2.0 * b.radius as f64)
            .fold(0.1f64, f64::max);
        let anchors: Vec<[f64; 3]> = bodies.iter().map(|b| b.pos).collect();
        Self {
            bodies,
            gravity: 9.81,
            ground_y: 0.0,
            substeps: 8,
            iterations: 8,
            margin: 5e-4,
            beta: 0.2,
            max_push: 3.0,
            angular_damping: 1.5,
            friction: 0.7,
            walls: None,
            mass_ratio_limit: 1000.0,
            sleep_linear: 0.05,
            sleep_time: 0.5,
            cell,
            contacts: Vec::new(),
            asleep: vec![false; n],
            sleep_for: vec![0.0; n],
            sleep_anchor: anchors,
            awake: n,
            warm: HashMap::new(),
            grid: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
            (p[2] / self.cell).floor() as i64,
        )
    }

    /// Обратная масса: спящее тело и пол — бесконечная масса.
    fn inv_mass_of(&self, i: usize) -> f64 {
        if i == GROUND || self.asleep[i] || self.bodies[i].mass <= 0.0 {
            0.0
        } else {
            1.0 / self.bodies[i].mass
        }
    }

    /// Обратные массы пары + правило `κ > limit`: тяжёлый партнёр становится
    /// бесконечномассовым для лёгкого, но его скорость читается.
    fn pair_inv_mass(&self, a: usize, b: usize) -> (f64, f64) {
        let mut wa = self.inv_mass_of(a);
        let mut wb = self.inv_mass_of(b);
        if a != GROUND && wa > 0.0 && wb > 0.0 {
            let (ma, mb) = (self.bodies[a].mass, self.bodies[b].mass);
            let ratio = (ma / mb).max(mb / ma);
            if ratio > self.mass_ratio_limit {
                if ma > mb {
                    wa = 0.0;
                } else {
                    wb = 0.0;
                }
            }
        }
        (wa, wb)
    }

    fn make_contact(&self, a: usize, b: usize, n: [f64; 3], pen: f64, kind: u8) -> PileContact {
        let (t1, t2) = tangents(n);
        let key = contact_key(a, b, kind);
        let (lambda_n, lambda_t1, lambda_t2) = self
            .warm
            .get(&key)
            .map(|w| (w[0], w[1], w[2]))
            .unwrap_or((0.0, 0.0, 0.0));
        PileContact {
            a,
            b,
            kind,
            n,
            pen,
            t1,
            t2,
            lambda_n,
            lambda_t1,
            lambda_t2,
        }
    }

    /// Применить импульсы прошлого подшага к скоростям до итераций.
    fn warm_start(&mut self) {
        for ci in 0..self.contacts.len() {
            let c = self.contacts[ci];
            if c.lambda_n <= 0.0 && c.lambda_t1 == 0.0 && c.lambda_t2 == 0.0 {
                continue;
            }
            let (wa, wb) = self.pair_inv_mass(c.a, c.b);
            if wa + wb <= 0.0 {
                continue;
            }
            let a_ground = c.a == GROUND;
            let ra_len = if a_ground {
                0.0
            } else {
                self.bodies[c.a].radius as f64
            };
            let rb = scale(c.n, -(self.bodies[c.b].radius as f64));
            let ra = scale(c.n, ra_len);
            let p = add(
                add(scale(c.n, c.lambda_n), scale(c.t1, c.lambda_t1)),
                scale(c.t2, c.lambda_t2),
            );
            if !a_ground && wa > 0.0 {
                let ia = self.bodies[c.a].inv_inertia;
                let a = &mut self.bodies[c.a];
                a.vel = sub(a.vel, scale(p, wa));
                a.ang = sub(a.ang, scale(cross(ra, p), ia));
            }
            let ib = self.bodies[c.b].inv_inertia;
            let b = &mut self.bodies[c.b];
            b.vel = add(b.vel, scale(p, wb));
            b.ang = add(b.ang, scale(cross(rb, p), ib));
        }
    }

    fn save_warm(&mut self) {
        self.warm.clear();
        for c in &self.contacts {
            if c.lambda_n <= 0.0 {
                continue;
            }
            self.warm
                .insert(contact_key(c.a, c.b, c.kind), [c.lambda_n, c.lambda_t1, c.lambda_t2]);
        }
    }

    fn build_grid(&mut self) {
        self.grid.clear();
        for i in 0..self.bodies.len() {
            let k = self.key(self.bodies[i].pos);
            self.grid.entry(k).or_default().push(i as u32);
        }
    }

    fn collect(&mut self) {
        self.contacts.clear();

        // пол
        for i in 0..self.bodies.len() {
            let b = &self.bodies[i];
            let pen = self.ground_y - (b.pos[1] - b.radius as f64);
            if pen > -self.margin {
                let c = self.make_contact(GROUND, i, [0.0, 1.0, 0.0], pen, 1);
                self.contacts.push(c);
            }
        }

        // стенки-ящик: нормаль внутрь, ось y открыта
        if let Some((lo, hi)) = self.walls {
            for i in 0..self.bodies.len() {
                let b = &self.bodies[i];
                let r = b.radius as f64;
                for axis in [0usize, 2] {
                    let left = lo[axis] - (b.pos[axis] - r);
                    if left > -self.margin {
                        let mut n = [0.0; 3];
                        n[axis] = 1.0;
                        let kind = if axis == 0 { 2 } else { 4 };
                        let c = self.make_contact(GROUND, i, n, left, kind);
                        self.contacts.push(c);
                    }
                    let right = (b.pos[axis] + r) - hi[axis];
                    if right > -self.margin {
                        let mut n = [0.0; 3];
                        n[axis] = -1.0;
                        let kind = if axis == 0 { 3 } else { 5 };
                        let c = self.make_contact(GROUND, i, n, right, kind);
                        self.contacts.push(c);
                    }
                }
            }
        }

        // пары — по сетке, с детерминированным обходом ключей
        let mut keys: Vec<(i64, i64, i64)> = self.grid.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let here = self.grid[&k].clone();
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let nk = (k.0 + dx, k.1 + dy, k.2 + dz);
                        let same_cell = nk == k;
                        if nk < k {
                            continue; // пару ячеек считаем один раз
                        }
                        let Some(other) = self.grid.get(&nk).map(|v| v.clone()) else {
                            continue;
                        };
                        for &ia in &here {
                            for &ib in &other {
                                let (i, j) = (ia as usize, ib as usize);
                                // В одной ячейке список обходится с самим собой —
                                // тут нужен порядок, чтобы не считать пару дважды.
                                // Между разными ячейками пара встречается ровно
                                // один раз, и порядок индексов произволен.
                                if same_cell && i >= j {
                                    continue;
                                }
                                if self.asleep[i] && self.asleep[j] {
                                    continue;
                                }
                                let (pa, pb) = (self.bodies[i].pos, self.bodies[j].pos);
                                let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                                let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                                let radii =
                                    self.bodies[i].radius as f64 + self.bodies[j].radius as f64;
                                if dist2 > radii * radii || dist2 == 0.0 {
                                    continue;
                                }
                                let dist = dist2.sqrt();
                                let n = [d[0] / dist, d[1] / dist, d[2] / dist];
                                let c = self.make_contact(i, j, n, radii - dist, 0);
                                self.contacts.push(c);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Пробуждение: спящего будит бодрствующий сосед, который реально едет.
    fn wake_pass(&mut self) {
        let mut to_wake: Vec<usize> = Vec::new();
        for c in &self.contacts {
            if c.a == GROUND {
                continue;
            }
            let (a, b) = (c.a, c.b);
            let sp = |s: &Self, i: usize| {
                let v = s.bodies[i].vel;
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            };
            if self.asleep[a] && !self.asleep[b] && sp(self, b) > 4.0 * self.sleep_linear {
                to_wake.push(a);
            } else if self.asleep[b] && !self.asleep[a] && sp(self, a) > 4.0 * self.sleep_linear {
                to_wake.push(b);
            }
        }
        for i in to_wake {
            self.asleep[i] = false;
            self.sleep_for[i] = 0.0;
            self.sleep_anchor[i] = self.bodies[i].pos;
        }
    }

    /// Скоростное решение контакта: нормаль с bias, затем трение по двум
    /// фиксированным касательным осям.
    fn solve(&mut self, ci: usize, h: f64) {
        let mut c = self.contacts[ci];
        let (wa, wb) = self.pair_inv_mass(c.a, c.b);
        let w = wa + wb;
        if w <= 0.0 {
            return;
        }
        let a_ground = c.a == GROUND;

        // радиусы-векторы от центров к точке контакта
        let ra_len = if a_ground {
            0.0
        } else {
            self.bodies[c.a].radius as f64
        };
        let rb_len = self.bodies[c.b].radius as f64;
        let ra = [c.n[0] * ra_len, c.n[1] * ra_len, c.n[2] * ra_len];
        let rb = [-c.n[0] * rb_len, -c.n[1] * rb_len, -c.n[2] * rb_len];

        let (va, wa_ang, ia) = if a_ground {
            ([0.0; 3], [0.0; 3], 0.0)
        } else {
            (
                self.bodies[c.a].vel,
                self.bodies[c.a].ang,
                self.bodies[c.a].inv_inertia,
            )
        };
        let vb = self.bodies[c.b].vel;
        let wb_ang = self.bodies[c.b].ang;
        let ib = self.bodies[c.b].inv_inertia;

        // скорость точки контакта
        let va_cp = add(va, cross(wa_ang, ra));
        let vb_cp = add(vb, cross(wb_ang, rb));
        let rel = sub(vb_cp, va_cp);

        // --- нормаль -------------------------------------------------------
        // Для сфер r ∥ n, поэтому нормальный импульс вращение не меняет.
        let vn = dot3(rel, c.n);
        let push = (((c.pen - self.margin).max(0.0) * self.beta) / h).min(self.max_push);
        let d_lambda = (push - vn) / w;
        let new_n = (c.lambda_n + d_lambda).max(0.0);
        let dl = new_n - c.lambda_n;
        c.lambda_n = new_n;
        if dl != 0.0 {
            if !a_ground && wa > 0.0 {
                for k in 0..3 {
                    self.bodies[c.a].vel[k] -= c.n[k] * wa * dl;
                }
            }
            let b = &mut self.bodies[c.b];
            for k in 0..3 {
                b.vel[k] += c.n[k] * wb * dl;
            }
        }

        // --- трение --------------------------------------------------------
        if self.friction > 0.0 && c.lambda_n > 0.0 {
            for axis in 0..2 {
                let t = if axis == 0 { c.t1 } else { c.t2 };
                // пересчитываем скорость точки контакта: нормаль её изменила
                let (va, wa_ang) = if a_ground {
                    ([0.0; 3], [0.0; 3])
                } else {
                    (self.bodies[c.a].vel, self.bodies[c.a].ang)
                };
                let vb = self.bodies[c.b].vel;
                let wb_ang = self.bodies[c.b].ang;
                let rel = sub(
                    add(vb, cross(wb_ang, rb)),
                    add(va, cross(wa_ang, ra)),
                );
                let vt = dot3(rel, t);

                let ra_xt = cross(ra, t);
                let rb_xt = cross(rb, t);
                let w_t = wa + wb + dot3(ra_xt, ra_xt) * ia + dot3(rb_xt, rb_xt) * ib;
                if w_t <= 0.0 {
                    continue;
                }
                let d_lambda_t = -vt / w_t;
                let cap = self.friction * c.lambda_n;
                let old = if axis == 0 { c.lambda_t1 } else { c.lambda_t2 };
                let new_t = (old + d_lambda_t).clamp(-cap, cap);
                let d = new_t - old;
                if axis == 0 {
                    c.lambda_t1 = new_t;
                } else {
                    c.lambda_t2 = new_t;
                }
                if d != 0.0 {
                    if !a_ground && wa > 0.0 {
                        let a = &mut self.bodies[c.a];
                        for k in 0..3 {
                            a.vel[k] -= t[k] * wa * d;
                            a.ang[k] -= ra_xt[k] * ia * d;
                        }
                    }
                    let b = &mut self.bodies[c.b];
                    for k in 0..3 {
                        b.vel[k] += t[k] * wb * d;
                        b.ang[k] += rb_xt[k] * ib * d;
                    }
                }
            }
        }

        self.contacts[ci] = c;
    }

    fn substep(&mut self, h: f64) {
        // 1. скорости и сопротивление качению
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.asleep[i] {
                continue;
            }
            b.vel[1] -= self.gravity * h;
            if b.inv_inertia > 0.0 && self.angular_damping > 0.0 {
                let f = (1.0 - self.angular_damping * h).max(0.0);
                for k in 0..3 {
                    b.ang[k] *= f;
                }
            }
        }

        // 2. контакты и скоростное решение
        self.build_grid();
        self.collect();
        self.wake_pass();
        self.warm_start();
        for _ in 0..self.iterations {
            for ci in 0..self.contacts.len() {
                self.solve(ci, h);
            }
        }
        self.save_warm();

        // 3. позиции и ориентация
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.asleep[i] {
                continue;
            }
            for k in 0..3 {
                b.pos[k] += b.vel[k] * h;
            }
            if b.inv_inertia > 0.0 {
                let w = b.ang;
                if w[0] != 0.0 || w[1] != 0.0 || w[2] != 0.0 {
                    integrate_quat(&mut b.rot, w, h);
                }
            }
        }

        // 4. сон по накопленному смещению
        let mut awake = 0usize;
        for i in 0..self.bodies.len() {
            if self.asleep[i] {
                continue;
            }
            let p = self.bodies[i].pos;
            let a = self.sleep_anchor[i];
            let d = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
            let moved = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if moved > self.sleep_linear * self.sleep_time {
                self.sleep_anchor[i] = p;
                self.sleep_for[i] = 0.0;
            } else {
                self.sleep_for[i] += h;
                if self.sleep_for[i] >= self.sleep_time {
                    self.asleep[i] = true;
                    self.bodies[i].vel = [0.0; 3];
                    self.bodies[i].ang = [0.0; 3];
                    self.sleep_anchor[i] = p;
                    continue;
                }
            }
            awake += 1;
        }
        self.awake = awake;
    }

    pub fn step(&mut self, dt: f64) {
        let h = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            self.substep(h);
        }
    }

    pub fn max_penetration(&mut self) -> f64 {
        self.build_grid();
        self.collect();
        let mut worst: f64 = 0.0;
        for c in &self.contacts {
            if c.pen > worst {
                worst = c.pen;
            }
        }
        worst
    }

    pub fn max_speed(&self) -> f64 {
        self.bodies
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.asleep[*i])
            .map(|(_, b)| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max)
    }

    pub fn top(&self) -> f64 {
        self.bodies
            .iter()
            .map(|b| b.pos[1] + b.radius as f64)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Демо-сцена D3: сферы радиусом 8 см в ящике, поставленные решёткой над полом.
///
/// Дрожание в позициях обязательно: при идеально ровной решётке каждая колонна
/// стоит вечно — нормаль строго вертикальна, касательной силы нет, трения нет.
/// Дрожание детерминированное, иначе тест перестал бы быть воспроизводимым.
pub fn pile_demo(count: usize) -> PileSim {
    let mut bodies = Vec::with_capacity(count);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / u32::MAX as f64) - 0.5
    };
    for i in 0..count.max(1) {
        // Решётка центрируется в ящике: иначе зазор слева и справа разный,
        // и куча после падения систематически смещается к ближней стенке.
        let x = 0.205 + (i % 17) as f64 * 0.24 + rnd() * 0.07;
        let z = 0.205 + ((i / 17) % 17) as f64 * 0.24 + rnd() * 0.07;
        let y = 1.0 + (i / 289) as f64 * 0.24 + rnd() * 0.02;
        let vx = rnd() * 0.06;
        let vz = rnd() * 0.06;
        bodies.push(Body::new([x, y, z], [vx, 0.0, vz], 1.0, 0.08));
    }
    let mut sim = PileSim::new(bodies);
    sim.gravity = 9.81;
    // Ящик: решётка занимает 0..4.08 по x и z, стенки чуть шире.
    sim.walls = Some(([-0.05, -1e9, -0.05], [4.2, 1e9, 4.2]));
    sim
}
