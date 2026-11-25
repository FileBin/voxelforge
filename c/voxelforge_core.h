#ifndef VOXELFORGE_CORE_H
#define VOXELFORGE_CORE_H

/* C-ядро voxelforge: устройство Vulkan, буферы, compute-пайплайны, запуск.
 *
 * Границы сознательные: здесь нет ни логики мира, ни физики, ни шейдерных
 * констант — только Vulkan. Всё, что можно назвать «игрой», живёт в Rust.
 * Шов односторонний: Rust вызывает C, обратных колбэков нет.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t code;      /* 0 — успех */
    const char *msg;   /* статическая строка, не освобождать */
} vf_status;

typedef struct vf_ctx vf_ctx;
typedef struct vf_buffer vf_buffer;
typedef struct vf_pipeline vf_pipeline;

/* Один dispatch: пайплайн, его storage-буферы по порядку привязки,
 * push-константы и число рабочих групп. */
typedef struct {
    vf_pipeline *pipeline;
    vf_buffer **buffers;
    uint32_t n_buffers;
    const void *push;
    uint32_t push_bytes;
    uint32_t groups[3];
} vf_dispatch;

vf_status vf_init(vf_ctx **out);
void vf_shutdown(vf_ctx *c);
vf_status vf_device_info(vf_ctx *c, char *name, size_t name_len,
                         uint32_t *api_version);

/* host_visible != 0 — память, доступная из CPU (для upload/download). */
vf_status vf_buffer_new(vf_ctx *c, size_t bytes, int host_visible,
                        vf_buffer **out);
void vf_buffer_free(vf_ctx *c, vf_buffer *b);
vf_status vf_buffer_write(vf_ctx *c, vf_buffer *b, size_t offset,
                          const void *src, size_t bytes);
vf_status vf_buffer_read(vf_ctx *c, vf_buffer *b, size_t offset, void *dst,
                         size_t bytes);

vf_status vf_pipeline_new(vf_ctx *c, const uint32_t *spirv, size_t spirv_bytes,
                          uint32_t n_storage_buffers, uint32_t push_bytes,
                          const char *debug_name, vf_pipeline **out);
void vf_pipeline_free(vf_ctx *c, vf_pipeline *p);

/* Записать все dispatch'и в один command buffer, отправить, дождаться.
 * Между dispatch'ами ставится барьер «запись шейдера → чтение шейдером»,
 * иначе следующий проход читает недописанные данные. */
vf_status vf_run(vf_ctx *c, vf_dispatch *list, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* VOXELFORGE_CORE_H */
