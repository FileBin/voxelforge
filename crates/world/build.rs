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
    // Живое окно (SDL3) собирается только с фичей `live`: без неё тестам
    // SDL и не нужен.
    let mut objs = vec![obj.clone()];
    if std::env::var("CARGO_FEATURE_LIVE").is_ok() {
        let view_src = Path::new("../../c/voxelforge_view.c");
        let view_obj = format!("{out}/voxelforge_view.o");
        let cflags = Command::new("pkg-config")
            .args(["--cflags", "sdl3"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let mut args: Vec<String> = vec![
            "-O2".into(),
            "-std=c11".into(),
            "-fPIC".into(),
            "-Wall".into(),
            "-c".into(),
            "-o".into(),
            view_obj.clone(),
            view_src.to_str().unwrap().into(),
        ];
        for f in cflags.split_whitespace() {
            args.push(f.to_string());
        }
        let st = Command::new("gcc").args(&args).status().expect("gcc");
        assert!(st.success(), "voxelforge_view.c не собрался");
        objs.push(view_obj);
        println!("cargo:rustc-link-lib=dylib=SDL3");
        println!("cargo:rerun-if-changed=../../c/voxelforge_view.c");
        println!("cargo:rerun-if-changed=../../c/voxelforge_view.h");
    }
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.clone()];
    ar_args.extend(objs.iter().cloned());
    let st = Command::new("ar").args(&ar_args).status().expect("ar");
    assert!(st.success(), "ar");

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=voxelforge_core");
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed=../../c/voxelforge_core.h");
}
