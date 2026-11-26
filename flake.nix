{
  description = "voxelforge — GPU-резидентный воксельный движок на решётке BCC";

  # Один вход — это и есть «работает через год»: flake.lock фиксирует точную
  # ревизию nixpkgs, из которой берутся rustc, glslc, spirv-tools, mesa и gcc.
  # flake.lock + Cargo.lock + `--locked` делают сборку воспроизводимой.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system (import nixpkgs { inherit system; }));

      toolchain = pkgs: with pkgs; [
        cargo
        rustc
        rustfmt
        clippy
        gcc
        cmake
        ninja
        pkg-config
        shaderc
        spirv-tools
        vulkan-headers
        vulkan-loader
        vulkan-tools
        sdl3
        # CPU-Vulkan (lavapipe) — headless-прогоны и CI, без касания GPU
        mesa
        wayland
        wayland-protocols
        libxkbcommon
      ];
    in
    {
      apps = forAll (system: pkgs:
        let
          engine = self.packages.${system}.default;

          help = pkgs.writeShellApplication {
            name = "voxelforge";
            text = ''
              echo "voxelforge — воксельный движок на решётке BCC"
              echo ""
              echo "  nix run .#world     мир: снаряд влетает в BCC-плиту, она раскалывается,"
              echo "                      осколки дробят ячейки 0.28 → 0.14 → 0.07 м"
              echo "  nix run .#viewer    локальный сервер вьюера (http://localhost:8080)"
              echo "  nix run .#ref       CPU-эталон (оракул для тестов), полный CLI"
              echo ""
              echo "  результат → demos/out/world.vfw; смотреть: nix run .#viewer"
              echo "  флаги движка: --frames --substeps --iterations --out --shaders --quiet"
            '';
          };

          viewer = pkgs.writeShellApplication {
            name = "voxelforge-viewer";
            runtimeInputs = [ pkgs.python3 ];
            text = ''
              if [ -f viewer/index.html ]; then
                root="$(pwd)"
              else
                root="${self}"
              fi
              echo "вьюер: http://localhost:8080/viewer/index.html"
              echo "мир:    http://localhost:8080/viewer/index.html?file=../demos/out/world.vfw"
              echo "Ctrl+C — остановить"
              exec python3 -m http.server 8080 --directory "$root"
            '';
          };

          # Живое окно: настоящая видеокарта, мышь и стрельба. Никакого
          # lavapipe — он не умеет показать окно.
          live = pkgs.writeShellApplication {
            name = "voxelforge-live";
            runtimeInputs = [ pkgs.vulkan-loader pkgs.sdl3 ];
            text = ''
              exec ${self.packages.${system}.live}/bin/voxelforge-live \
                --shaders ${self.packages.${system}.spirv} "$@"
            '' + ''
            '';
          };

          # Мир: движок считает на видеокарте. В CI остаётся lavapipe.
          world = pkgs.writeShellApplication {
            name = "voxelforge-world";
            runtimeInputs = [ pkgs.mesa pkgs.vulkan-loader ];
            text = ''
              out="''${OUT_DIR:-demos/out}"
              mkdir -p "$out"
              if [ -z "''${VK_ICD_FILENAMES:-}" ]; then
                lvp="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -n1)"
                [ -n "$lvp" ] && export VK_ICD_FILENAMES="$lvp"
              fi
              ${engine}/bin/voxelforge \
                --shaders ${self.packages.${system}.spirv} \
                --out "$out/world.vfw" "$@"
              echo ""
              echo "смотреть: nix run .#viewer → перетащить $out/world.vfw"
            '';
          };
        in
        {
          default = { type = "app"; program = "${help}/bin/voxelforge"; };
          help = { type = "app"; program = "${help}/bin/voxelforge"; };
          world = { type = "app"; program = "${world}/bin/voxelforge-world"; };
          viewer = { type = "app"; program = "${viewer}/bin/voxelforge-viewer"; };
          live = { type = "app"; program = "${live}/bin/voxelforge-live"; };
          ref = { type = "app"; program = "${engine}/bin/vf-ref"; };
        });

      packages = forAll (system: pkgs: {
        # SPIR-V из GLSL: собирается один раз и передаётся движку путём.
        spirv = pkgs.runCommand "voxelforge-spv" {
          nativeBuildInputs = [ pkgs.shaderc ];
        } ''
          mkdir -p "$out"
          cd ${./shaders}
          for f in *.comp; do
            glslc -I. "$f" -o "$out/''${f%.comp}.spv"
          done
          for f in view_*.vert view_*.frag; do
            glslc -O -o "$out/''${f%.*}.''${f##*.}.spv" "$f"
          done
        '';

        default = pkgs.rustPlatform.buildRustPackage {
          pname = "voxelforge";
          version = "0.0.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let base = baseNameOf (toString path);
              in !(builtins.elem base [ "target" "result" "demos" ".direnv" ]);
          };
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.gcc ];
          # headers нужны на этапе сборки C-ядра (build.rs зовёт gcc),
          # loader — на линковке и в рантайме
          buildInputs = [ pkgs.vulkan-headers pkgs.vulkan-loader ];
          meta.mainProgram = "voxelforge";
        };

        # Живое окно: тот же движок плюс SDL3-слой и графические конвейеры.
        live = pkgs.rustPlatform.buildRustPackage {
          pname = "voxelforge-live";
          version = "0.0.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.; 
            filter = path: type:
              let base = baseNameOf (toString path);
              in !(builtins.elem base [ "target" "result" "demos" ".direnv" ]);
          };
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [ "--features" "live" ];
          doCheck = false;
          nativeBuildInputs = [ pkgs.gcc pkgs.pkg-config ];
          buildInputs = [ pkgs.vulkan-headers pkgs.vulkan-loader pkgs.sdl3 ];
          meta.mainProgram = "voxelforge-live";
        };
      });

      checks = forAll (system: pkgs: {
        # Юнит- и интеграционные тесты: BCC-решётка, раскол, подразбиение,
        # физика эталона, сверка CPU и GPU, вода.
        tests = self.packages.${system}.default.overrideAttrs (_: {
          doCheck = true;
          # тестам раскола нужны SPIR-V: без них проверка пропускается
          preCheck = ''
            export VF_SHADERS=${self.packages.${system}.spirv}
            export VK_ICD_FILENAMES="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json | head -n1)"
          '';
        });

        # Мир целиком на lavapipe: движок обязан отработать headless и выдать
        # запись, в которой осколки действительно раздробились (форм > тел).
        demo-world = pkgs.runCommand "demo-world" { } ''
          mkdir -p "$out"
          export VK_ICD_FILENAMES="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json | head -n1)"
          ${self.packages.${system}.default}/bin/voxelforge \
            --shaders ${self.packages.${system}.spirv} \
            --frames 60 --out "$out/world.vfw" | tee "$out/log.txt"
          test -s "$out/world.vfw"
          grep -q "раскалывается на" "$out/log.txt"
          echo "мир записан: $(stat -c%s "$out/world.vfw") байт"
        '';

        # Эталон: орбита замыкается, энергия не дрейфует.
        demo-orbit = pkgs.runCommand "demo-orbit" { } ''
          mkdir -p "$out"
          ${self.packages.${system}.default}/bin/vf-ref \
            --scenario orbit --frames 600 --periods 2 --substeps 4 \
            --json "$out/orbit.json" --out "$out/orbit.vfr" | tee "$out/log.txt"
        '';

        # D1 и D2: контакты. Падают, если проникновение > 2 см,
        # остаточная скорость > 2 см/с или стопка просела больше 5%.
        demo-stack = pkgs.runCommand "demo-stack" { } ''
          mkdir -p "$out"
          ${self.packages.${system}.default}/bin/vf-ref \
            --scenario stack --bodies 10 --frames 600 --substeps 8 --iterations 8 \
            --json "$out/stack.json" --out "$out/stack.vfr" | tee "$out/log.txt"
        '';
      });

      devShells = forAll (system: pkgs: {
        default = pkgs.mkShell {
          name = "voxelforge";
          packages = toolchain pkgs;
          nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook ];
          RUST_BACKTRACE = "1";
          shellHook = ''
            # lavapipe: полностью программный Vulkan. Прогоны по умолчанию идут
            # на нём — он не трогает видеокарту и не может её подвесить.
            if [ -z "''${VK_ICD_FILENAMES:-}" ]; then
              lvp="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -n1)"
              if [ -n "$lvp" ]; then export VK_ICD_FILENAMES="$lvp"; fi
            fi
            echo "voxelforge · rustc $(rustc --version | cut -d' ' -f2) · nixpkgs ${builtins.substring 0 7 nixpkgs.rev}"
            [ -n "''${VK_ICD_FILENAMES:-}" ] && echo "  lavapipe: $VK_ICD_FILENAMES"
            echo "  шейдеры: nix build .#spirv -o shaders/spv-out"
            echo "  тесты:   cargo test --locked"
            echo "  мир:     nix run .#world   (или ./target/release/voxelforge --help)"
          '';
        };
      });
    };
}
