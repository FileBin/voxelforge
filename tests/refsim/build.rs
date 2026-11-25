use std::path::Path;
use std::process::Command;

/// Собирает C-ядро статической библиотекой и линкует с ней тестовый крейт.
///
/// Без `cc`-крейта сознательно: у проекта нет ни одной внешней зависимости, и
/// `cargo build --locked` должен работать в офлайне. Хватит gcc и ar.
/// Статика, а не .so, чтобы бинарник запускался откуда угодно без rpath.
fn main() {
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = Path::new("../../c/voxelforge_core.c");
    let obj = format!("{out}/voxelforge_core.o");
    let lib = format!("{out}/libvoxelforge_core.a");

    let run = |cmd: &str, args: &[&str]| {
        let st = Command::new(cmd)
            .args(args)
            .status()
            .unwrap_or_else(|e| panic!("{cmd} не запустился: {e} (он есть в dev shell)"));
        assert!(st.success(), "{cmd} завершился с ошибкой");
    };

    run(
        "gcc",
        &[
            "-O2",
            "-std=c11",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-c",
            "-o",
            &obj,
            src.to_str().unwrap(),
        ],
    );
    run("ar", &["rcs", &lib, &obj]);

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=voxelforge_core");
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed=../../c/voxelforge_core.h");
}
