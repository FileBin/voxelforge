{
  description = "voxelforge — GPU-резидентный воксельный движок (BCC, C-ядро + Rust-хост)";

  # Один вход — это и есть «работает через год»: flake.lock фиксирует точную
  # ревизию nixpkgs, из которой берутся rustc, glslc, spirv-tools, mesa и gcc.
  # Комбинация flake.lock + Cargo.lock + `--locked` делает сборку
  # воспроизводимой независимо от того, что успело уехать вперёд на crates.io.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system (import nixpkgs { inherit system; }));

      # Общие пакеты тулчейна: и для devShell, и для будущей сборки C-ядра.
      toolchain = pkgs: with pkgs; [
        # Rust
        cargo
        rustc
        rustfmt
        clippy
        # линковка тестов и будущего C-ядра (на хосте нет cc в голом шелле)
        gcc
        # C-ядро (появится в M0)
        cmake
        ninja
        pkg-config
        # Vulkan: компилятор шейдеров, валидаторы, интроспекция
        shaderc
        spirv-tools
        vulkan-headers
        vulkan-loader
        vulkan-tools
        # CPU-Vulkan (lavapipe) — headless-прогоны и CI, без касания GPU
        mesa
        # окружение поверхности окна (понадобится с M0)
        wayland
        wayland-protocols
        libxkbcommon
      ];
    in
    {
      apps = forAll (system: pkgs:
        let
          vfref = self.packages.${system}.default;

          # Обёртка демки: пишет результаты в $OUT_DIR (по умолчанию ./out),
          # лишние аргументы после `--` уходят прямо в vf-ref и перекрывают
          # дефолты (парсер берёт последнее значение).
          demo = name: scenario: defaults: pkgs.writeShellApplication {
            name = "voxelforge-${name}";
            text = ''
              out="''${OUT_DIR:-demos/out}"
              mkdir -p "$out"
              ${vfref}/bin/vf-ref --scenario ${scenario} ${defaults} \
                --out "$out/${name}.vfr" --svg "$out/${name}.svg" --json "$out/${name}.json" \
                "$@"
              echo ""
              echo "готово: $out/${name}.vfr"
              echo "смотреть: открыть viewer/index.html и перетащить туда файл"
            '';
          };

          help = pkgs.writeShellApplication {
            name = "voxelforge";
            text = ''
              echo "voxelforge — демки физики на CPU-эталоне"
              echo ""
              echo "  nix run .#orbit       орбиты: звезда + 3 планеты, 2 периода"
              echo "  nix run .#collapse    гравитационный коллапс облака (400 тел)"
              echo "  nix run .#vf-ref -- --scenario orbit --frames 900   прямой доступ к CLI"
              echo "  nix run .#help        эта справка"
              echo ""
              echo "  результаты → ./out  (другой каталог: OUT_DIR=dir nix run .#orbit)"
              echo "  просмотр   → viewer/index.html, перетащить .vfr в окно"
              echo ""
              echo "  все флаги: --frames --dt --periods --substeps --bodies --seed --svg --size"
            '';
          };
        in
        {
          default = { type = "app"; program = "${help}/bin/voxelforge"; };
          help = { type = "app"; program = "${help}/bin/voxelforge"; };
          orbit = {
            type = "app";
            program = "${demo "orbit" "orbit" "--frames 1500 --periods 2 --substeps 4"}/bin/voxelforge-orbit";
          };
          collapse = {
            type = "app";
            program = "${demo "collapse" "collapse" "--bodies 400 --frames 900 --seed 7 --dt 0.01"}/bin/voxelforge-collapse";
          };
          vf-ref = { type = "app"; program = "${vfref}/bin/vf-ref"; };
        });

      devShells = forAll (system: pkgs: {
        default = pkgs.mkShell {
          name = "voxelforge";
          packages = toolchain pkgs;
          # Даёт LIBCLANG_PATH, если в проекте когда-нибудь появится bindgen.
          nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook ];
          RUST_BACKTRACE = "1";
          shellHook = ''
            # lavapipe: полностью программный Vulkan. Демки и тесты по умолчанию
            # гоняются на нём — он не трогает видеокарту и не может её подвесить.
            if [ -z "''${VK_ICD_FILENAMES:-}" ]; then
              lvp="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -n1)"
              if [ -n "$lvp" ]; then export VK_ICD_FILENAMES="$lvp"; fi
            fi
            echo "voxelforge · rustc $(rustc --version | cut -d' ' -f2) · nixpkgs ${builtins.substring 0 7 nixpkgs.rev}"
            [ -n "''${VK_ICD_FILENAMES:-}" ] && echo "  lavapipe: $VK_ICD_FILENAMES"
            echo "  тесты:  cargo test --locked"
            echo "  демо:   cargo run --release --locked -p voxelforge-refsim --bin vf-ref -- --scenario orbit --out demos/out/orbit.vfr"
          '';
        };
      });

      packages = forAll (system: pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "vf-ref";
          version = "0.0.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let base = baseNameOf (toString path);
              in !(builtins.elem base [ "target" "result" "demos" ".direnv" ]);
          };
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.gcc ];
          meta.mainProgram = "vf-ref";
        };
      });

      checks = forAll (system: pkgs: {
        # Один шлюз: юнит-тесты эталона (замыкание орбиты, дрейф энергии,
        # сохранение импульса, побайтовый детерминизм записи).
        tests = self.packages.${system}.default.overrideAttrs (_: { doCheck = true; });

        # Демка как приёмочный тест: падает, если дрейф энергии > 1%
        # или орбита не замкнулась за 1%.
        demo-orbit = pkgs.runCommand "demo-orbit" { } ''
          mkdir -p "$out"
          ${self.packages.${system}.default}/bin/vf-ref \
            --scenario orbit --frames 600 --periods 2 --substeps 4 \
            --json "$out/orbit.json" --out "$out/orbit.vfr" | tee "$out/log.txt"
        '';
      });
    };
}
