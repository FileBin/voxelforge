//! Логика мира voxelforge: BCC-воксели, тела, симуляция на GPU, запись мира.
//!
//! Движок один и работает только с решёткой BCC:
//! - `bcc` — ячейка (усечённый октаэдр), уровни, подразбиение, раскол;
//! - `types` — тело для солвера (масса, форма, ориентация);
//! - `sim` — Vulkan-солвер через C-ядро (`c/voxelforge_core.c`);
//! - `vfw` — формат `.vfw`, который читает вьюер.
//!
//! Кубических вокселей здесь нет: ячейка — ячейка Вороного решётки BCC.

pub mod bcc;
pub mod field;
pub mod sdf_contact;
pub mod collide;
pub mod live;
pub mod sim;
pub mod t1;
pub mod tet;
pub mod types;
pub mod vfw;
