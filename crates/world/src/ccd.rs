//! Непрерывное обнаружение столкновений (CCD) бисекцией по SDF.
//!
//! Зачем: обычный контакт ищется только в позах на концах шага. Если тело за
//! шаг смещается дальше, чем его собственный размер, оно может оказаться по
//! другую сторону партнёра, ни разу не попав внутрь на момент проверки —
//! «туннелирование». Лечение: для таких пар ищем ПЕРВЫЙ момент касания вдоль
//! перемещения за шаг (бисекция) и ставим тело в эту позу; импульс считается
//! там же, как у обычного контакта.
//!
//! Всё по полям расстояний, без геометрии тетов:
//!  * пробные точки — узлы поля ВНУТРИ тела, у каждой запомнена глубина под
//!    собственной поверхностью `d` (метры);
//!  * условие касания — `f_other(p) < d`, то есть поверхность пробного тела
//!    дошла до поверхности партнёра (а не «внутренняя точка утонула»);
//!  * бисекция: `t=0.5`; касание есть → `[0, 0.5]`, нет → `[0.5, 1]`;
//!    12 итераций дают ~1e-3 шага перемещения.
//!
//! Порог включения — ОТНОСИТЕЛЬНОЕ смещение за шаг больше размера меньшего
//! тела (`needs_continuous`): медленные тела идут прежним путём и ничего не
//! платят.

use crate::field::NodeField;

/// Сколько пробных точек берём с тела. Точки — узлы у ПОВЕРХНОСТИ (сортируем
/// по глубине): именно они дают точный момент касания.
pub const PROBE_POINTS: usize = 256;

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Кватернион (w, x, y, z) — как в `common.glsl`/`live::rot`.
#[inline]
pub fn quat_rot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[1], q[2], q[3]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[0])), cross(u, t))
}

#[inline]
pub fn quat_inv_rot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rot([q[0], -q[1], -q[2], -q[3]], v)
}

/// Точка внутри тела: положение в системе тела (от центроида) и глубина под
/// собственной поверхностью, метры.
#[derive(Clone, Copy, Debug)]
pub struct ProbePoint {
    pub local: [f32; 3],
    pub depth: f32,
}

/// Тело глазами CCD: запечённое поле, поза, скорость, инерция.
pub struct BodyView<'a> {
    pub field: &'a NodeField,
    /// Центр поля относительно центроида в системе тела.
    pub shift: [f32; 3],
    pub pos: [f32; 3],
    pub quat: [f32; 4],
    pub vel: [f32; 3],
    pub ang: [f32; 3],
    pub inv_mass: f32,
    pub inv_inertia: f32,
    /// Полуразмер тела (метры) — порог «смещается больше своего размера».
    pub radius: f32,
    pub asleep: bool,
}

impl<'a> BodyView<'a> {
    /// Значение поля (метры) в мировой точке при заданной позе тела.
    fn value_at(&self, pos: [f32; 3], w: [f32; 3]) -> f32 {
        let p = quat_inv_rot(self.quat, sub(w, pos));
        let n = [
            (p[0] - self.shift[0]) / self.field.step,
            (p[1] - self.shift[1]) / self.field.step,
            (p[2] - self.shift[2]) / self.field.step,
        ];
        self.field.sample_m(n)
    }

    /// Единичная нормаль поверхности (мир) в мировой точке.
    fn normal_at(&self, pos: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        let p = quat_inv_rot(self.quat, sub(w, pos));
        let n = [
            (p[0] - self.shift[0]) / self.field.step,
            (p[1] - self.shift[1]) / self.field.step,
            (p[2] - self.shift[2]) / self.field.step,
        ];
        quat_rot(self.quat, self.field.gradient(n))
    }
}

/// Пробные точки тела: узлы поля внутри, ближайшие к поверхности — первыми.
pub fn probe_points(f: &NodeField, shift: [f32; 3], want: usize) -> Vec<ProbePoint> {
    let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
    let mut all: Vec<ProbePoint> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let v = f.val[(z * ny + y) * nx + x];
                if !(v < 0.0) {
                    continue;
                }
                let n = [
                    (f.lo[0] + x as i32) as f32,
                    (f.lo[1] + y as i32) as f32,
                    (f.lo[2] + z as i32) as f32,
                ];
                all.push(ProbePoint {
                    local: [
                        shift[0] + n[0] * f.step,
                        shift[1] + n[1] * f.step,
                        shift[2] + n[2] * f.step,
                    ],
                    depth: -v * f.step,
                });
            }
        }
    }
    /* У поверхности поле точнее: точка с глубиной 0.5 шага показывает момент
     * касания с ошибкой не больше этой глубины (она вычитается в условии).
     * Берём ПЕРВЫЕ по глубине (самые близкие к поверхности) и прореживаем их:
     * прореживать весь список нельзя — он отсортирован по глубине, и шаг по
     * нему дотягивается до центра тела (это давало ложное «уже пересеклись»). */
    all.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));
    let want = want.max(1);
    let cap = (want * 8).min(all.len());
    if cap <= want {
        all.truncate(cap);
        return all;
    }
    let stride = (cap / want).max(1);
    all[..cap].iter().step_by(stride).take(want).copied().collect()
}

/// Включать ли continuous для пары: относительное смещение за шаг больше
/// размера меньшего тела.
pub fn needs_continuous(rel_speed: f32, dt: f32, radius: f32) -> bool {
    rel_speed * dt > 2.0 * radius
}

/// Первое касание вдоль шага.
#[derive(Clone, Copy, Debug)]
pub struct SweepHit {
    /// Доля шага, где поверхности впервые соприкоснулись.
    pub t: f32,
    /// Точка касания (мир).
    pub p: [f32; 3],
    /// Нормаль от пробного тела к партнёру.
    pub n: [f32; 3],
    /// Заглубление на найденной позе, метры (может быть чуть больше нуля).
    pub depth: f32,
    /// Сколько итераций бисекции реально понадобилось до допуска.
    pub iters: u32,
}

/// Допуск бисекции — доля перемещения за шаг.
pub const T_TOL: f32 = 1e-3;
/// Потолок шагов сферного трейсинга (обычно хватает 4–8).
const TRACE_STEPS: u32 = 48;
/// Проб в грубом равномерном проходе, если трейсинг не сошёлся.
const SCAN_STEPS: u32 = 48;

/// Ищет первую точку касания поверхности `probe` с полем `other` на отрезке
/// перемещения за `dt`. Возвращает `None`, если касания по пути нет.
pub fn first_contact(
    pts: &[ProbePoint],
    probe: &BodyView,
    other: &BodyView,
    dt: f32,
    max_iters: u32,
) -> Option<SweepHit> {
    if pts.is_empty() || dt <= 0.0 {
        return None;
    }
    /* Глубочайший «заход» поверхности за шаг s: min по точкам (f_other - d).
     * < 0 — поверхности пересеклись. */
    let probe_pos = |s: f32| add(probe.pos, scale(probe.vel, dt * s));
    let other_pos = |s: f32| add(other.pos, scale(other.vel, dt * s));
    let deepest = |s: f32| -> Option<(f32, ProbePoint, [f32; 3])> {
        let pp = probe_pos(s);
        let op = other_pos(s);
        let mut best: Option<(f32, ProbePoint, [f32; 3])> = None;
        for pt in pts.iter() {
            let w = add(pp, quat_rot(probe.quat, pt.local));
            let over = other.value_at(op, w) - pt.depth;
            if best.map(|(b, _, _)| over < b).unwrap_or(true) {
                best = Some((over, *pt, w));
            }
        }
        best
    };

    let (over0, _, _) = deepest(0.0)?;
    if over0 < 0.0 {
        return None; /* уже в контакте — этим занимается обычный проход */
    }

    /* КАК ИЩЕМ ОТРЕЗОК. Тело за шаг может пройти НАСКВОЗЬ: в конце шага
     * касания снова нет, поэтому «делить пополам» от t=1 нельзя — нужен
     * сначала отрезок, где пересечение ЕСТЬ. Проходим путь сферным трейсингом
     * по полю (шаг = расстоянию до поверхности партнёра: пропустить её
     * невозможно), пока не окажемся внутри, — это и есть отрезок для бисекции.
     * Если трейсинг не сошёлся (скольжение вдоль поверхности), берём грубый
     * равномерный проход: без отрезка continuous просто не включается. */
    let path = norm(sub(probe.vel, other.vel)) * dt;
    let mut bracket = None;
    if path > 1e-9 {
        let mut s = 0.0f32;
        let mut prev = 0.0f32;
        for _ in 0..TRACE_STEPS {
            let (g, _, _) = deepest(s)?;
            if g < 0.0 {
                bracket = Some((prev, s));
                break;
            }
            /* Шаг в долях пути: g метров — гарантированный запас до касания. */
            let ds = (g.max(1e-3) / path).max(1e-6);
            prev = s;
            s += ds;
            if s >= 1.0 {
                break;
            }
        }
    }
    if bracket.is_none() {
        /* равномерный проход: 48 проб по пути, потом бисекция в найденном
         * отрезке (пропустить очень тонкое тело он может — тогда CCD просто
         * не сработает и останется прежнее поведение) */
        let mut prev = 0.0f32;
        for k in 1..=SCAN_STEPS {
            let s = k as f32 / SCAN_STEPS as f32;
            let (g, _, _) = deepest(s)?;
            if g < 0.0 {
                bracket = Some((prev, s));
                break;
            }
            prev = s;
        }
    }
    let Some((mut lo, mut hi)) = bracket else {
        return None; /* за шаг поверхности не встретились */
    };

    let (mut used, max_iters) = (0u32, max_iters.clamp(1, 24));
    while hi - lo > T_TOL && used < max_iters {
        let mid = 0.5 * (lo + hi);
        let (over, _, _) = deepest(mid)?;
        if over < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        used += 1;
    }

    /* Контакт считаем на позе t=hi (первое найденное проникновение).
     *
     * Нормаль и точка — с поверхности ПАРТНЁРА, в точке, куда упёрся центр
     * пробного тела (проекция по полю, 4 шага Ньютона). Так удар шара в грань
     * даёт нормаль ГРАНИ и плечо вдоль неё — без паразитного вращения, которое
     * получалось, если брать случайную внутреннюю точку: у неё нормаль
     * «своего» поля наклонена, и шар начинал крутиться от прямого удара. */
    let (over, _, w) = deepest(hi)?;
    let op = other_pos(hi);
    /* Нормаль берём в ПРОБНОЙ точке (она уже у поверхности партнёра, где поле
     * определено), а не в центре пробного тела: центр может лежать вне сетки
     * поля, и там все выборки зажаты — градиент нулевой (был ровный ноль). */
    let mut g = other.normal_at(op, w);
    if norm(g) < 0.5 {
        let d = sub(w, op);
        g = if norm(d) > 1e-6 { scale(d, 1.0 / norm(d)) } else { [0.0, 1.0, 0.0] };
    }
    /* Точка касания — центр пробного тела, спроецированный на поверхность
     * партнёра ВДОЛЬ нормали: плечо получается вдоль удара, без паразитного
     * вращения от случайной точки. */
    let mut q = probe_pos(hi);
    for _ in 0..4 {
        let v = other.value_at(op, q);
        q = sub(q, scale(g, v));
    }
    let n = scale(g, -1.0);
    Some(SweepHit {
        t: hi,
        p: q,
        n,
        depth: (-over).max(0.0),
        iters: used,
    })
}

/// Импульс одного контакта — та же математика, что в `solve.comp`
/// (`dl = max(push - vn, 0)/wn`, деления на число контактов нет: контакт один).
#[derive(Clone, Copy, Debug, Default)]
pub struct Impulse {
    pub dv_a: [f32; 3],
    pub dw_a: [f32; 3],
    pub dv_b: [f32; 3],
    pub dw_b: [f32; 3],
    /// Относительная нормальная скорость до импульса (м/с, минус — сближение).
    pub vn: f32,
}

pub fn contact_impulse(a: &BodyView, b: &BodyView, hit: &SweepHit) -> Impulse {
    let ra = sub(hit.p, a.pos);
    let rb = sub(hit.p, b.pos);
    let va = add(a.vel, cross(a.ang, ra));
    let vb = add(b.vel, cross(b.ang, rb));
    let vn = dot(sub(vb, va), hit.n);
    if vn >= 0.0 || a.asleep && b.asleep {
        return Impulse { vn, ..Default::default() };
    }
    let ran = cross(ra, hit.n);
    let rbn = cross(rb, hit.n);
    let wn = a.inv_mass + b.inv_mass + dot(ran, ran) * a.inv_inertia + dot(rbn, rbn) * b.inv_inertia;
    if wn <= 0.0 {
        return Impulse { vn, ..Default::default() };
    }
    /* push = 0: к моменту касания сближение уже недопустимо, поэтому
     * нормальная скорость гасится целиком (жёсткий удар, без отскока). */
    let dl = -vn / wn;
    let p = scale(hit.n, dl);
    Impulse {
        dv_a: scale(p, -a.inv_mass),
        dw_a: scale(cross(ra, p), -a.inv_inertia),
        dv_b: scale(p, b.inv_mass),
        dw_b: scale(cross(rb, p), b.inv_inertia),
        vn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet::TetBody;

    fn view<'a>(b: &'a TetBody, pos: [f32; 3], vel: [f32; 3], radius: f32) -> BodyView<'a> {
        BodyView {
            field: b.sdf(),
            shift: [0.0; 3],
            pos,
            quat: [1.0, 0.0, 0.0, 0.0],
            vel,
            ang: [0.0; 3],
            inv_mass: 1.0 / 8.0,
            inv_inertia: 1.0 / 0.06,
            radius,
            asleep: false,
        }
    }

    /// Бисекция находит момент касания шара с плитой: аналитика для шара r=0.14,
    /// летящего в плиту 1.12 м с фронтом в x=-0.61 на 240 м/с за 50 мс.
    #[test]
    fn bisection_finds_first_touch() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        assert!(pts.len() > 32, "пробных точек мало: {}", pts.len());

        let speed = 240.0f32;
        let dt = 0.05f32;
        let from = [-3.0, 0.0, 0.0];
        let a = view(&ball, from, [speed, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(needs_continuous(speed, dt, a.radius), "240 м/с обязан включить CCD");

        let hit = first_contact(&pts, &a, &b, dt, 12).expect("касание обязано найтись");
        /* аналитика: (3 - 0.56 - 0.14)/12 = 0.1917 */
        let want = (3.0 - 0.56 - 0.14) / 12.0;
        println!(
            "t={:.4} (аналитика {want:.4}) · глубина {:.4} м · итераций {} · n={:?}",
            hit.t, hit.depth, hit.iters, hit.n
        );
        assert!((hit.t - want).abs() < 0.01, "t={} вместо {want}", hit.t);
        assert!(hit.depth < 0.05, "глубина на позе касания {:.4} м", hit.depth);
        assert!(hit.n[0] > 0.9, "нормаль должна смотреть по движению: {:?}", hit.n);
    }

    /// Промах (шар летит выше плиты) касания не даёт — CCD не срабатывает.
    #[test]
    fn bisection_reports_no_hit_on_miss() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        let a = view(&ball, [-3.0, 2.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(first_contact(&pts, &a, &b, 0.05, 12).is_none());
    }

    /// Порог: медленное тело continuous не включает (поведение как раньше).
    #[test]
    fn continuous_gate_uses_own_size() {
        assert!(!needs_continuous(5.0, 0.05, 0.14), "5 м/с * 50 мс = 0.25 < 0.28");
        assert!(needs_continuous(240.0, 0.05, 0.14));
        /* уже в контакте — CCD не нужен, обычный проход справится */
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        let a = view(&ball, [0.0, 0.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(first_contact(&pts, &a, &b, 0.05, 12).is_none());
    }

    /// Импульс гасит нормальную скорость и сохраняет импульс пары.
    #[test]
    fn impulse_kills_normal_velocity_and_conserves_momentum() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let a = view(&ball, [-0.7, 0.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let mut b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        b.inv_mass = 1.0 / 590.0;
        b.inv_inertia = 1.0 / 60.0;
        let hit = SweepHit {
            t: 0.2,
            p: [-0.61, 0.0, 0.0],
            n: [1.0, 0.0, 0.0],
            depth: 0.001,
            iters: 12,
        };
        let im = contact_impulse(&a, &b, &hit);
        let va = add(a.vel, im.dv_a);
        let vb = add(b.vel, im.dv_b);
        println!("vn={:.2} · снаряд {:.2} м/с · плита {:.3} м/с", im.vn, va[0], vb[0]);
        /* после удара тела движутся вместе: относительная нормальная скорость 0,
         * а не «снаряд остановился в лабораторной системе» (импульс общий) */
        assert!((vb[0] - va[0]).abs() < 0.5, "сближение не погашено: {:.2}", vb[0] - va[0]);
        assert!(va[0] < 10.0, "снаряд должен потерять почти всю скорость: {:.2}", va[0]);
        assert!(vb[0] > 0.1, "плита обязана получить импульс: {:.3}", vb[0]);
        /* импульс: суммарный до и после совпадает (8 кг * 240 м/с) */
        let mom = 8.0 * va[0] + 590.0 * vb[0];
        assert!((mom - 8.0 * 240.0).abs() < 2.0, "импульс не сохранился: {mom:.3}");
    }
}
