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
 * push-константы и число рабочих групп. `stage` — произвольный номер стадии,
 * который хост хочет видеть в диагностике производительности (ядро его не
 * толкует, только возвращает вместе с таймстемпами). */
typedef struct {
    vf_pipeline *pipeline;
    vf_buffer **buffers;
    uint32_t n_buffers;
    const void *push;
    uint32_t push_bytes;
    uint32_t groups[3];
    uint32_t stage;
} vf_dispatch;

vf_status vf_init(vf_ctx **out);

/* Ядро владеет девайсом: окно работает на нём же, поэтому буферы тел,
 * контактов и геометрии видны и compute, и графике. inst_ext — расширения
 * instance, нужные окну (VK_KHR_surface + wayland/xcb); windowed != 0 просит
 * семейство очереди с графикой и включает VK_KHR_swapchain. */
vf_status vf_init_ext(vf_ctx **out, const char **inst_ext, uint32_t n_ext,
                      int windowed);

/* Для окна: instance с VK_KHR_surface и расширением оконной системы (какие
 * есть в драйвере), девайс с графикой и swapchain. */
vf_status vf_init_for_window(vf_ctx **out);

/* Девайс наружу — для окна (шаг 0: один Vulkan-девайс на всё). */
void *vf_ctx_instance(vf_ctx *c);
void *vf_ctx_phys(vf_ctx *c);
void *vf_ctx_device(vf_ctx *c);
void *vf_ctx_queue(vf_ctx *c);
uint32_t vf_ctx_family(vf_ctx *c);

/* Очередь одна на всех: submit'ы из разных потоков обязаны быть
 * взаимоисключающими. Ждать fence нужно ВНЕ замка. */
/* Дождаться, пока GPU закончит ВСЁ. Обязательно перед освобождением буферов
 * и девайса: такт отправляется асинхронно и может ещё считаться. */
vf_status vf_idle(vf_ctx *c);

void vf_ctx_lock(vf_ctx *c);
void vf_ctx_unlock(vf_ctx *c);

void vf_shutdown(vf_ctx *c);
vf_status vf_device_info(vf_ctx *c, char *name, size_t name_len,
                         uint32_t *api_version);

/* host_visible != 0 — память, доступная из CPU (для upload/download). */
vf_status vf_buffer_new(vf_ctx *c, size_t bytes, int host_visible,
                        vf_buffer **out);
void vf_buffer_free(vf_ctx *c, vf_buffer *b);
/* Сам VkBuffer (void*) — окну, чтобы читать буфер тел на том же девайсе. */
void *vf_buffer_handle(vf_buffer *b);
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

/* То же, но без ожидания: такт уходит, CPU идёт дальше. Результат такого
 * такта читается следующим — после vf_wait_back(c, 1). */
vf_status vf_run_async(vf_ctx *c, vf_dispatch *list, uint32_t count);
vf_status vf_wait_back(vf_ctx *c, uint32_t back);

/* --- диагностика производительности: GPU-таймстемпы по диспатчам такта ---
 *
 * Когда сбор включён, в тот же command buffer между диспатчами пишутся
 * vkCmdWriteTimestamp (count+1 метка: до первого и после каждого), а в конце
 * они копируются в host-visible буфер. Внутри такта НИЧЕГО не ждётся, поэтому
 * включение не сериализует пайплайн: результат забирается на следующем такте,
 * после fence. Выключено по умолчанию — при выключенном сборе в command buffer
 * не попадает ни одной лишней команды.
 *
 * `vf_perf_supported` — 1, если драйвер умеет таймстемпы compute-очереди
 * (timestampComputeAndGraphics && timestampValidBits > 0 && period > 0).
 * Попытка включить при неподдерживаемом драйвере возвращает ошибку и печатает
 * причину в stderr: молча «нулевых» чисел не бывает. */
int vf_perf_supported(vf_ctx *c);
vf_status vf_perf_enable(vf_ctx *c, int on);
int vf_perf_on(vf_ctx *c);

/* Результаты последнего ЗАКРЫТОГО такта (back = 0 — только что отправленный,
 * 1 — предыдущий). Заполняет ms[k] (мс) и stage[k] (номер стадии из
 * vf_dispatch.stage) для первых max диспатчей, *tick_ms — время всего такта от
 * первой метки до последней. Возвращает число диспатчей или -1, если данных
 * нет (сбор выключен, такт ещё не закрыт или слот уже переиспользован). */
int vf_perf_read(vf_ctx *c, uint32_t back, float *ms, uint32_t *stage, int max,
                 float *tick_ms);

#ifdef __cplusplus
}
#endif

#endif /* VOXELFORGE_CORE_H */
