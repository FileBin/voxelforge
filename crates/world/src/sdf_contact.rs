//! Контакты SDF↔SDF: CPU-оракул алгоритма, который затем переносится в шейдер.
//!
//! Старт — НЕ центр AABB (у тора там пусто), а узлы сетки ОДНОГО тела рядом с
//! пересечением AABB обоих: годен узел, где `f_a + f_b < marg`. Сумма знаковых
//! расстояний — это зазор между телами в этой точке: отрицательная внутри
//! пересечения, ноль на касании, положительная в зазоре. Такой старт
//! топологически нейтрален (тор, зацепления, U-формы) и находится ДО
//! проникновения, чего прежнее условие «оба поля < 0» не умело: у шара r=0.14
//! с шагом поля 0.14 внутри ровно один узел, и он попадал внутрь партнёра
//! только при перекрытии на радиус — на 20 м/с аппарат пролетал это окно за
//! один подшаг, и контактов не было ВООБЩЕ.
//!
//! Затем 4 итерации спуска по градиентам обоих полей. Глубина — ЗНАКОВАЯ:
//! `-(f_a + f_b)`; отрицательная (зазор) — законный спекулятивный контакт,
//! солвер ограничивает по нему сближение.

use crate::field::NodeField;

/// Один контакт: точка, нормаль (от A к B) и глубина проникновения (метры).
/// Глубина отрицательна, когда тела ещё не коснулись (спекулятивный контакт).
#[derive(Clone, Copy, Debug)]
pub struct SdfContact {
    pub p: [f32; 3],
    pub n: [f32; 3],
    pub depth: f32,
}

fn to_world(f: &NodeField, origin: [f32; 3], p: [f32; 3]) -> [f32; 3] {
    [
        origin[0] + p[0] * f.step,
        origin[1] + p[1] * f.step,
        origin[2] + p[2] * f.step,
    ]
}

fn to_node(f: &NodeField, origin: [f32; 3], p: [f32; 3]) -> [f32; 3] {
    [
        (p[0] - origin[0]) / f.step,
        (p[1] - origin[1]) / f.step,
        (p[2] - origin[2]) / f.step,
    ]
}

fn bounds(f: &NodeField, o: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let lo = to_world(f, o, [f.lo[0] as f32, f.lo[1] as f32, f.lo[2] as f32]);
    let hi = to_world(
        f,
        o,
        [
            (f.lo[0] + f.dims[0] as i32 - 1) as f32,
            (f.lo[1] + f.dims[1] as i32 - 1) as f32,
            (f.lo[2] + f.dims[2] as i32 - 1) as f32,
        ],
    );
    (lo, hi)
}

/// Один скан: затравки — узлы поля `fs` (поза `os`), партнёр — `ft`.
/// Возвращает число узлов, прошедших предварительный отбор (годных затравок).
fn scan(
    fs: &NodeField,
    os: [f32; 3],
    ft: &NodeField,
    ot: [f32; 3],
    marg: f32,
    max_contacts: usize,
    out: &mut Vec<SdfContact>,
) -> usize {
    let (alo, ahi) = bounds(fs, os);
    let (blo, bhi) = bounds(ft, ot);
    /* пересечение расширено на маржу: у грубых полей боксы могут не
     * пересечься даже когда поверхности уже почти касаются */
    let lo = [
        alo[0].max(blo[0] - marg),
        alo[1].max(blo[1] - marg),
        alo[2].max(blo[2] - marg),
    ];
    let hi = [
        ahi[0].min(bhi[0] + marg),
        ahi[1].min(bhi[1] + marg),
        ahi[2].min(bhi[2] + marg),
    ];
    if lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2] {
        return 0; // AABB не пересекаются
    }

    let nb = [fs.dims[0], fs.dims[1], fs.dims[2]];
    let mut seeds = 0usize;
    for zi in 0..nb[2] {
        for yi in 0..nb[1] {
            for xi in 0..nb[0] {
                let p_a = [
                    (fs.lo[0] + xi as i32) as f32,
                    (fs.lo[1] + yi as i32) as f32,
                    (fs.lo[2] + zi as i32) as f32,
                ];
                let w = to_world(fs, os, p_a);
                if w[0] < lo[0] || w[0] > hi[0] || w[1] < lo[1] || w[1] > hi[1] || w[2] < lo[2] || w[2] > hi[2] {
                    continue;
                }
                let fa_v = fs.val[(zi * nb[1] + yi) * nb[0] + xi];
                let ft_v = ft.sample(to_node(ft, ot, w));
                /* зазор в этой точке: сумма знаковых расстояний (метры) */
                let gap0 = fa_v * fs.step + ft_v * ft.step;
                if gap0 >= marg {
                    continue;
                }
                seeds += 1;
                /* спуск к паре ближайших точек поверхностей; шаг ограничен
                 * половиной шага узла — иначе на глубоком перекрытии точка
                 * улетает из пересечения за одну итерацию */
                let mut p = w;
                let lim = 0.5 * fs.step;
                for _ in 0..4 {
                    let pa = to_node(fs, os, p);
                    let pt = to_node(ft, ot, p);
                    let ga = fs.gradient(pa);
                    let gt = ft.gradient(pt);
                    let va = fs.sample(pa) * fs.step;
                    let vt = ft.sample(pt) * ft.step;
                    let mut d = [0.0f32; 3];
                    for k in 0..3 {
                        d[k] = 0.5 * (va * ga[k] + vt * gt[k]);
                    }
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if len > lim {
                        for k in 0..3 {
                            d[k] *= lim / len;
                        }
                    }
                    for k in 0..3 {
                        p[k] -= d[k];
                    }
                }
                let pa = to_node(fs, os, p);
                let pt = to_node(ft, ot, p);
                let va = fs.sample(pa) * fs.step;
                let vt = ft.sample(pt) * ft.step;
                let gap = va + vt;
                if gap >= marg {
                    continue;
                }
                /* Спуск ищет пару БЛИЖАЙШИХ точек поверхностей: у двух выпуклых
                 * тел это окружность пересечения поверхностей, где сумма = 0, а
                 * настоящая глубина — глубже. Берём БОЛЬШУЮ глубину из узла-
                 * затравки и спущенной точки (то же правило в шейдере). */
                let (pp, ga, gt) = if gap < gap0 {
                    (p, fs.gradient(pa), ft.gradient(pt))
                } else {
                    let qa = to_node(fs, os, w);
                    let qt = to_node(ft, ot, w);
                    (w, fs.gradient(qa), ft.gradient(qt))
                };
                let gap = if gap < gap0 { gap } else { gap0 };
                let p = pp;
                let mut n = [ga[0] - gt[0], ga[1] - gt[1], ga[2] - gt[2]];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
                n = [n[0] / len, n[1] / len, n[2] / len];
                /* дедуп: контакт ближе шага к уже найденному — тот же контакт */
                if out.iter().any(|c| {
                    let d = [c.p[0] - p[0], c.p[1] - p[1], c.p[2] - p[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < (fs.step * 1.5) * (fs.step * 1.5)
                }) {
                    continue;
                }
                /* глубина — ЗНАКОВАЯ: плюс внутри, минус в зазоре */
                out.push(SdfContact { p, n, depth: -gap });
            }
        }
    }
    seeds
}

/// Контакты между полями `fa`, `fb` (позиции — мировые центры тел).
///
/// `max_contacts` ограничивает число контактов на пару: их избыток на больших
/// перекрытиях стоит дороже, чем тет-пары.
pub fn contacts(
    fa: &NodeField,
    oa: [f32; 3],
    fb: &NodeField,
    ob: [f32; 3],
    max_contacts: usize,
) -> Vec<SdfContact> {
    /* маржа = шаг более мелкой сетки: затравки дискретны, точность контакта
     * ограничена шагом (в шейдере к ней добавляется путь за подшаг) */
    let marg = fa.step.min(fb.step);
    let mut out: Vec<SdfContact> = Vec::new();
    /* затравки — по более мелкому полю: узлов у поверхности там больше */
    let (seed_a, seeds) = if fa.step <= fb.step {
        (true, scan(fa, oa, fb, ob, marg, max_contacts, &mut out))
    } else {
        (false, scan(fb, ob, fa, oa, marg, max_contacts, &mut out))
    };
    /* симметрия: в этой сетке пересечения может не быть вовсе — пробуем из
     * второго тела */
    if seeds == 0 {
        if seed_a {
            scan(fb, ob, fa, oa, marg, max_contacts, &mut out);
        } else {
            scan(fa, oa, fb, ob, marg, max_contacts, &mut out);
        }
    }
    /* порядок — от самых глубоких: на мелком перекрытии первые попадания
     * обхода оказывались мелкими касаниями у края пятна */
    out.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(max_contacts);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet::TetBody;

    /// Два шара радиуса 0.3 с центрами на расстоянии 0.5: перекрытие 0.1 м.
    /// Глубина контакта обязана быть 0.1 в пределах половины шага узла.
    #[test]
    fn two_spheres_depth_matches_analytic() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.3, 0);
        let b = TetBody::ball([0.5, 0.0, 0.0], 0.3, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        println!(
            "контактов {}, глубины {:?}",
            cs.len(),
            cs.iter().map(|c| c.depth).collect::<Vec<_>>()
        );
        assert!(!cs.is_empty(), "перекрытие 0.1 м должно дать контакты");
        let max_d = cs.iter().fold(0.0f32, |m, c| m.max(c.depth));
        /* точность глубины — шаг узла: она складывается из значений поля в
         * двух узлах, каждое с погрешностью до половины шага */
        assert!((max_d - 0.1).abs() < a.sdf().step, "глубина {max_d} вместо 0.1");
    }

    /// Разнесённые шары (зазор 1.4 м) контактов не дают.
    #[test]
    fn separate_spheres_have_no_contacts() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.3, 0);
        let b = TetBody::ball([2.0, 0.0, 0.0], 0.3, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        assert!(cs.is_empty(), "контактов быть не должно: {}", cs.len());
    }

    /// ШАРЫ r=0.14 НА КАСАНИИ (центры 0.28): сетка шара — 0.14 м, внутри один
    /// узел (центр). Прежнее условие «оба поля < 0» не находило тут ничего, и
    /// на 20 м/с шар проходил сквозь шар. Контакт обязан быть, и глубина —
    /// около нуля, а не «нет контакта».
    #[test]
    fn touching_small_spheres_give_contact() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let b = TetBody::ball([0.28, 0.0, 0.0], 0.14, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        assert!(!cs.is_empty(), "на касании контакт обязан быть");
        let max_d = cs.iter().fold(f32::MIN, |m, c| m.max(c.depth));
        assert!(max_d.abs() < a.sdf().step, "глубина на касании {max_d}, ждали ~0");
    }

    /// Зазор 5 мм у мелких шаров (в пределах маржи = шаг узла): контакт
    /// СПЕКУЛЯТИВНЫЙ (глубина < 0) — солвер по нему не пускает тела насквозь,
    /// но и не толкает. Именно этого не хватало: без спекулятивной маржи
    /// контакт появлялся только при перекрытии на радиус, и на 20 м/с тело
    /// пролетало это окно за один подшаг.
    #[test]
    fn near_touch_small_spheres_is_speculative() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let b = TetBody::ball([0.295, 0.0, 0.0], 0.14, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        let near: Vec<f32> = cs.iter().map(|c| c.depth).filter(|d| *d < 0.0).collect();
        assert!(!near.is_empty(), "зазор 5 мм должен дать спекулятивный контакт: {:?}",
            cs.iter().map(|c| c.depth).collect::<Vec<_>>());
    }
}
