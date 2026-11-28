//! Контакты SDF↔SDF: CPU-оракул алгоритма, который затем переносится в шейдер.
//!
//! Старт — НЕ центр AABB (у тора там пусто), а узлы сетки внутри пересечения
//! объёмов: `f_a(p) < 0 && f_b(p) < 0`. Затем 3–5 итераций спуска по градиентам
//! обоих полей. Такой старт топологически нейтрален: тор, зацепления, U-формы
//! обрабатываются одинаково.

use crate::field::NodeField;

/// Один контакт: точка, нормаль (от A к B) и глубина проникновения (метры).
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
    let mut out: Vec<SdfContact> = Vec::new();
    /* перебор узлов сетки A внутри пересечения AABB */
    let na = [fa.dims[0], fa.dims[1], fa.dims[2]];
    let pa_lo = to_world(fa, oa, [fa.lo[0] as f32, fa.lo[1] as f32, fa.lo[2] as f32]);
    let pa_hi = to_world(
        fa,
        oa,
        [
            (fa.lo[0] + na[0] as i32 - 1) as f32,
            (fa.lo[1] + na[1] as i32 - 1) as f32,
            (fa.lo[2] + na[2] as i32 - 1) as f32,
        ],
    );
    let nb = [fb.dims[0], fb.dims[1], fb.dims[2]];
    let pb_lo = to_world(fb, ob, [fb.lo[0] as f32, fb.lo[1] as f32, fb.lo[2] as f32]);
    let pb_hi = to_world(
        fb,
        ob,
        [
            (fb.lo[0] + nb[0] as i32 - 1) as f32,
            (fb.lo[1] + nb[1] as i32 - 1) as f32,
            (fb.lo[2] + nb[2] as i32 - 1) as f32,
        ],
    );
    let lo = [pa_lo[0].max(pb_lo[0]), pa_lo[1].max(pb_lo[1]), pa_lo[2].max(pb_lo[2])];
    let hi = [pa_hi[0].min(pb_hi[0]), pa_hi[1].min(pb_hi[1]), pa_hi[2].min(pb_hi[2])];
    if lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2] {
        return out; // AABB не пересекаются
    }

    let mut best_fallback: Option<([f32; 3], f32)> = None;
    for zi in 0..nb[2] {
        for yi in 0..nb[1] {
            for xi in 0..nb[0] {
                let p_b = [
                    (fb.lo[0] + xi as i32) as f32,
                    (fb.lo[1] + yi as i32) as f32,
                    (fb.lo[2] + zi as i32) as f32,
                ];
                let w = to_world(fb, ob, p_b);
                if w[0] < lo[0] || w[0] > hi[0] || w[1] < lo[1] || w[1] > hi[1] || w[2] < lo[2] || w[2] > hi[2] {
                    continue;
                }
                let fb_v = fb.val[(zi * nb[1] + yi) * nb[0] + xi];
                let pa = to_node(fa, oa, w);
                let fa_v = fa.sample(pa);
                /* вырожденный случай: запоминаем лучший «заход» на случай, если
                 * ни одного узла внутри обоих тел не окажется */
                let score = fa_v.max(fb_v);
                if best_fallback.map(|(_, s)| score < s).unwrap_or(true) {
                    best_fallback = Some((w, score));
                }
                if fa_v >= 0.0 || fb_v >= 0.0 {
                    continue;
                }
                /* спуск к паре ближайших точек поверхностей */
                let mut p = w;
                for _ in 0..4 {
                    let pa = to_node(fa, oa, p);
                    let pb = to_node(fb, ob, p);
                    let ga = fa.gradient(pa);
                    let gb = fb.gradient(pb);
                    let va = fa.sample(pa) * fa.step;
                    let vb = fb.sample(pb) * fb.step;
                    for k in 0..3 {
                        p[k] -= 0.5 * (va * ga[k] + vb * gb[k]);
                    }
                }
                let pa = to_node(fa, oa, p);
                let pb = to_node(fb, ob, p);
                let va = fa.sample(pa) * fa.step;
                let vb = fb.sample(pb) * fb.step;
                if va >= 0.0 || vb >= 0.0 {
                    continue;
                }
                let ga = fa.gradient(pa);
                let gb = fb.gradient(pb);
                let mut n = [ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
                n = [n[0] / len, n[1] / len, n[2] / len];
                /* дедуп: контакт ближе шага к уже найденному — тот же контакт */
                if out.iter().any(|c| {
                    let d = [c.p[0] - p[0], c.p[1] - p[1], c.p[2] - p[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < (fa.step * 1.5) * (fa.step * 1.5)
                }) {
                    continue;
                }
                /* глубина пары — сумма заглублений обеих сторон: сдвинуть тела
                * надо на неё, а не на половину */
                out.push(SdfContact { p, n, depth: (-va) + (-vb) });
                if out.len() >= max_contacts {
                    return out;
                }
            }
        }
    }
    /* ничего внутри — берём точку наибольшего захода (тонкое касание) */
    if out.is_empty() {
        if let Some((w, score)) = best_fallback {
            if score < fa.step * 0.5 {
                let pa = to_node(fa, oa, w);
                let pb = to_node(fb, ob, w);
                let ga = fa.gradient(pa);
                let gb = fb.gradient(pb);
                let mut n = [ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
                n = [n[0] / len, n[1] / len, n[2] / len];
                out.push(SdfContact { p: w, n, depth: 0.0 });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet::TetBody;

    /// Два шара радиуса 0.3 с центрами на расстоянии 0.5: перекрытие 0.1 м.
    /// Глубина контакта обязана быть 0.1 в пределах четверти шага узла.
    #[test]
    fn two_spheres_depth_matches_analytic() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.3, 0);
        let b = TetBody::ball([0.5, 0.0, 0.0], 0.3, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        println!("контактов {}, глубины {:?}", cs.len(), cs.iter().map(|c| c.depth).collect::<Vec<_>>());
        assert!(!cs.is_empty(), "перекрытие 0.1 м должно дать контакты");
        let max_d = cs.iter().fold(0.0f32, |m, c| m.max(c.depth));
        assert!((max_d - 0.1).abs() < a.sdf().step * 0.5, "глубина {max_d} вместо 0.1");
    }

    /// Разнесённые шары контактов не дают.
    #[test]
    fn separate_spheres_have_no_contacts() {
        let a = TetBody::ball([0.0, 0.0, 0.0], 0.3, 0);
        let b = TetBody::ball([2.0, 0.0, 0.0], 0.3, 0);
        let cs = contacts(a.sdf(), a.origin, b.sdf(), b.origin, 16);
        assert!(cs.is_empty(), "контактов быть не должно: {}", cs.len());
    }
}
