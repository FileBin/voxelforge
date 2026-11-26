#!/usr/bin/env bash
# Headless-прогон живого окна — как scripts/headless-run.sh в SandboxGame:
# виртуальный Wayland-композитор (sway, WLR_BACKENDS=headless) + программный
# Vulkan (Mesa lavapipe). Настоящая видеокарта и рабочий стол не трогаются,
# но путь кода тот же, что у тебя в окне: SDL3 → surface → swapchain →
# графические конвейеры → кадры.
#
#   scripts/headless-run.sh [бинарник] [кадров] [кадры выстрелов]
#
# Печатает сводку, которую читает flake check:
#   HEADLESS exit=0 errors=0 swapchain=1 frames=N cracks=M bodies=K
set +e

BIN="${1:-target/release/voxelforge-live}"
FRAMES="${2:-180}"
SHOTS="${3:-20,60,120}"
SECONDS_LIMIT="${HEADLESS_SECONDS:-90}"

if [ ! -x "$BIN" ]; then
    echo "headless: '$BIN' не исполняемый" >&2
    exit 2
fi

WORK="$(mktemp -d)"
export XDG_RUNTIME_DIR="$WORK/runtime"
export HOME="$WORK/home"
mkdir -p "$XDG_RUNTIME_DIR" "$HOME"
chmod 700 "$XDG_RUNTIME_DIR"

export WLR_BACKENDS=headless
export WLR_RENDERER=pixman
export WLR_LIBINPUT_NO_DEVICES=1
export SDL_VIDEODRIVER=wayland
export SDL_VIDEO_DRIVER=wayland
export XDG_SESSION_TYPE=wayland

# Программный Vulkan: явный ICD выигрывает у глоба.
if [ -z "${VK_DRIVER_FILES:-}" ]; then
    if [ -n "${VK_ICD_FILENAMES:-}" ] && [ -f "$VK_ICD_FILENAMES" ]; then
        export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
    else
        # самый свежий lvp в сторе, а не первый попавшийся
        icd="$(ls -t /nix/store/*-mesa-*/share/vulkan/icd.d/lvp_icd.x86_64.json 2>/dev/null | head -n1)"
        [ -n "$icd" ] && export VK_DRIVER_FILES="$icd"
    fi
fi

SWAY_BIN="${SWAY_BIN:-sway}"
"$SWAY_BIN" -c /dev/null >"$WORK/sway.log" 2>&1 &
SWAY_PID=$!

unset WAYLAND_DISPLAY
DETECTED=""
for _ in $(seq 1 100); do
    for sock in "$XDG_RUNTIME_DIR"/wayland-[0-9]*; do
        case "$sock" in *.lock) continue ;; esac
        if [ -S "$sock" ]; then DETECTED="$(basename "$sock")"; break; fi
    done
    [ -n "$DETECTED" ] && break
    kill -0 "$SWAY_PID" 2>/dev/null || break
    sleep 0.1
done

if [ -z "$DETECTED" ]; then
    echo "headless: композитор не поднялся" >&2
    tail -5 "$WORK/sway.log" >&2
    kill -9 "$SWAY_PID" 2>/dev/null
    exit 3
fi
export WAYLAND_DISPLAY="$DETECTED"

for sock in "$XDG_RUNTIME_DIR"/sway-ipc.*.sock; do
    [ -S "$sock" ] && export SWAYSOCK="$sock" && break
done

printf 'headless: композитор %s, %s кадров, выстрелы на %s\n' "$WAYLAND_DISPLAY" "$FRAMES" "$SHOTS"

CPUS="${HEADLESS_CPUS:-}"
if [ -z "$CPUS" ] && command -v taskset >/dev/null 2>&1; then
    TOTAL="$(nproc)"
    LIMIT=$(( TOTAL > 4 ? TOTAL - 3 : TOTAL ))
    CPUS="0-$(( LIMIT - 1 ))"
fi
[ -n "$CPUS" ] && echo "headless: cpu mask $CPUS"

SHADER_ARG=""
[ -n "${VF_SHADERS:-}" ] && SHADER_ARG="--shaders $VF_SHADERS"

# HEADLESS_GDB=1 — прогнать под отладчиком и напечатать backtrace.
RUNNER=""
if [ -n "${HEADLESS_GDB:-}" ]; then
    RUNNER="gdb -batch -ex run -ex bt -ex 'info registers rip' --args"
fi

# shellcheck disable=SC2086
( exec ${CPUS:+taskset -c $CPUS} $RUNNER "$BIN" $SHADER_ARG --frames "$FRAMES" --shots "$SHOTS" ${HEADLESS_EXTRA:-} ) \
    >"$WORK/live.log" 2>&1 &
APP_PID=$!

# HEADLESS_SHOT=/path.png — снять кадр с композитора (нужен grim) и потом
# закрыть приложение. Полезно, чтобы посмотреть, что реально нарисовано.
if [ -n "${HEADLESS_SHOT:-}" ]; then
    for _ in $(seq 1 60); do
        kill -0 "$APP_PID" 2>/dev/null || break
        sleep 0.1
    done
    sleep 1
    grim "$HEADLESS_SHOT" >"$WORK/grim.log" 2>&1 || echo "headless: grim не снял кадр: $(cat "$WORK/grim.log")" >&2
    [ -s "$HEADLESS_SHOT" ] && echo "headless: кадр $HEADLESS_SHOT"
    kill -9 "$APP_PID" 2>/dev/null
    kill -9 "$SWAY_PID" 2>/dev/null
    exit 0
fi

# Ждём штатного выхода по числу кадров; если приложение упало — выходим сразу.
for _ in $(seq 1 $(( SECONDS_LIMIT * 10 ))); do
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

EXIT_CODE=0
if kill -0 "$APP_PID" 2>/dev/null; then
    EXIT_CODE=124
    kill -9 "$APP_PID" 2>/dev/null
else
    wait "$APP_PID"
    EXIT_CODE=$?
fi
kill -9 "$SWAY_PID" 2>/dev/null

ERRORS=$(grep -c '\[ERROR\]' "$WORK/live.log")
SWAPCHAIN=$(grep -c '^\[view\]' "$WORK/live.log")
FRAMES_DONE=$(sed -n 's/.*кадров \([0-9]*\).*/\1/p' "$WORK/live.log" | tail -1)
CRACKS=$(sed -n 's/.*трещин \([0-9]*\).*/\1/p' "$WORK/live.log" | tail -1)
BODIES=$(sed -n 's/.*тел \([0-9]*\) ·.*/\1/p' "$WORK/live.log" | tail -1)

echo "--- лог приложения (хвост) ---"
tail -14 "$WORK/live.log"
echo "--- сводка ---"
echo "HEADLESS exit=$EXIT_CODE errors=$ERRORS view=$SWAPCHAIN frames=${FRAMES_DONE:-0} cracks=${CRACKS:-0} bodies=${BODIES:-0}"
echo "LOG_DIR=$WORK"

# 139 = SIGSEGV, 134 = SIGABRT: это отдельный, «плохой» случай.
if [ "$EXIT_CODE" -eq 139 ] || [ "$EXIT_CODE" -eq 134 ]; then
    echo "headless: ПАДЕНИЕ (signal $(( EXIT_CODE - 128 )))" >&2
    exit 5
fi
[ "$EXIT_CODE" -eq 0 ] && [ "$ERRORS" -eq 0 ] && [ "${FRAMES_DONE:-0}" -gt 0 ]
