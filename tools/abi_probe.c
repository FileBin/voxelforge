/* Сверка раскладки структур между Rust и C: печатает sizeof/offsetof.
 * Собирается и запускается вручную (не входит в cargo build):
 *   gcc -std=c11 -I c -o /tmp/abi tools/abi_probe.c && /tmp/abi
 */
#include <stddef.h>
#include <stdio.h>

#include "voxelforge_view.h"

#define P(T, f) printf("%-14s %-16s %3zu\n", #T, #f, offsetof(T, f))
#define S(T) printf("%-14s sizeof           %3zu  align %3zu\n", #T, sizeof(T), _Alignof(T))

int main(void) {
    S(vf_perf_row);
    P(vf_perf_row, name);
    P(vf_perf_row, min_ms);
    P(vf_perf_row, mean_ms);
    P(vf_perf_row, max_ms);
    P(vf_perf_row, share);

    S(vf_scene_counts);
    P(vf_scene_counts, total);
    P(vf_scene_counts, awake);
    P(vf_scene_counts, contacts);

    S(vf_rates);
    P(vf_rates, fps_mean);
    P(vf_rates, fps_min);
    P(vf_rates, fps_max);
    P(vf_rates, tps_mean);
    P(vf_rates, tps_min);
    P(vf_rates, tps_max);
    P(vf_rates, frame_mean_ms);
    P(vf_rates, frame_min_ms);
    P(vf_rates, frame_max_ms);
    P(vf_rates, tick_mean_ms);
    P(vf_rates, tick_min_ms);
    P(vf_rates, tick_max_ms);
    P(vf_rates, frames);
    P(vf_rates, ticks);
    P(vf_rates, frame_hist);
    P(vf_rates, frame_hist_n);
    P(vf_rates, tick_hist);
    P(vf_rates, tick_hist_n);
    return 0;
}
