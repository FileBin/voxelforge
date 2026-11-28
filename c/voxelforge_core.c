/* libvoxelforge_core — Vulkan core, plain C, C ABI. См. voxelforge_core.h.
 *
 * Только Vulkan: нет логики мира, нет физики, нет шейдерных констант.
 * Модель исполнения простая и достаточная для headless-демок: записали пачку
 * dispatch'ей в один command buffer, отправили, дождались fence, прочитали
 * результат из host-visible памяти. Никаких фреймов, swapchain и синхронизации
 * с окном — это появится вместе с окном, а не раньше.
 */

/* clock_gettime/CLOCK_MONOTONIC: -std=c11 без этого прячет POSIX-объявления в
 * glibc, и сборка падает на «implicit declaration». Определять ДО первого
 * системного заголовка. */
#define _POSIX_C_SOURCE 199309L

#include "voxelforge_core.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

/* Настенное время CPU в миллисекундах: им меряются фазы кадра и такта
 * (запись команд, замок, submit, fence). GPU-время — отдельно, таймстемпами. */
static double vf_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}


#define VF_MAX_BINDINGS 16
/* Сколько тактов может быть в полёте. 3 — с запасом: CPU успевает прочитать
 * статистику прошлого такта, пока считаются следующие один-два. */
#define VF_FRAMES 3
#define VF_MAX_SETS 8192
/* Потоки submit'ов: такт (физика) и кадр (рендер) — у каждого своя очередь,
 * кольцо command buffer'ов, fence'ы и таймлайн-семафор. */
#define VF_TICK 0
#define VF_FRAME 1
#define VF_STREAMS 2
/* Сколько таймлайнов можно ждать одним submit'ом. */
#define VF_MAX_WAITS 4
/* Потолок диспатчей на такт для диагностики стадий. Реальный такт сейчас
 * укладывается в ~700 (куски × подшаги × итерации), 2048 — с запасом. */
#define VF_PERF_MAX 2048

#define OK ((vf_status){0, "ok"})
#define ERR(code, text) ((vf_status){(code), (text)})

struct vf_buffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    size_t bytes;
    void *mapped;
};

struct vf_pipeline {
    VkPipeline pipe;
    VkPipelineLayout layout;
    VkDescriptorSetLayout dsl;
    uint32_t n_buffers;
    uint32_t push_bytes;
};

struct vf_pass {
    VkCommandPool pool;
    VkCommandBuffer cb[VF_FRAMES];
    VkFence fence[VF_FRAMES];
    /* Таймлайн-семафор потока: значение = номер submit'а в этом потоке.
     * Бинарный семафор говорит «готово», таймлайн — «готово до такта N»;
     * кадру нужно именно второе: он читает банк, опубликованный тактом N,
     * и не обязан ждать такт N+1. */
    VkSemaphore tl;
    uint64_t submitted;
    /* Пул дескрипторов НА СЛОТ кольца: сброс пула, чей command buffer ещё
     * исполняется, — это освобождение памяти дескрипторов под работающим
     * шейдером (на lavapipe это SIGSEGV в handle_descriptor_sets). Слот
     * сбрасывается только после того, как его fence закрылся. */
    VkDescriptorPool desc_pool[VF_FRAMES];
    /* Стадии последнего submit'а этого потока (CPU, мс): запись команд,
     * ожидание замка очереди, сам vkQueueSubmit и ожидание fence (режим
     * «ждать сразу»). Хост читает их из таблицы диагностики. */
    double st_record, st_lock, st_submit, st_fence;
};

struct vf_ctx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice dev;
    /* Очередь кадра — graphics, приоритет 1.0 (её же берёт окно: present
     * обязан поддерживаться её семейством). */
    VkQueue queue;
    uint32_t family;
    /* Очередь такта — compute, приоритет 0.0. На RADV это ДРУГОЕ семейство
     * (4 очереди, без графики), то есть отдельное железное кольцо; если
     * такого семейства нет — та же очередь, что у кадра (split = 0). */
    VkQueue tick_queue;
    uint32_t tick_family;
    int split;
    float prio_gfx;
    float prio_tick;
    /* Значение кадрового таймлайна, дочитавшего банк b (см. vf_note_bank_read). */
    uint64_t bank_read[VF_BANKS];
    /* Потоки submit'ов: VF_TICK — физика, VF_FRAME — кадр. У каждого свой
     * пул команд, кольцо fence'ов, таймлайн и пул дескрипторов: два потока
     * пишут их из разных потоков CPU. */
    struct vf_pass q[VF_STREAMS];
    VkPhysicalDeviceMemoryProperties memp;
    /* Диагностика стадий такта (VF_PERF / F3). Пул запросов СВОЙ на слот
     * кольца: сбрасывать общий пул, пока предыдущий такт ещё пишет в него
     * метки, — гонка; со своим слот не переиспользуется до fence. */
    VkQueryPool qpool[VF_FRAMES];
    vf_buffer *tsbuf[VF_FRAMES];
    uint32_t perf;        /* сбор включён хостом */
    uint32_t perf_ok;     /* драйвер умеет таймстемпы на этой очереди */
    uint32_t perf_warned; /* про переполнение VF_PERF_MAX уже сообщили */
    float period_ms;      /* timestampPeriod (нс) -> мс на тик */
    uint32_t ts_bits;     /* timestampValidBits семейства очереди */
    int timestamp_cg;     /* limits.timestampComputeAndGraphics */
    /* Что именно отправлено в слот: число диспатчей, их номера стадий и
     * индекс submit'а. Хост-сторона заполняет при отправке, читает после
     * fence — данные GPU для этого не нужны. */
    uint32_t perf_n[VF_FRAMES];
    uint32_t perf_stage[VF_FRAMES][VF_PERF_MAX];
    uint64_t perf_id[VF_FRAMES];
    /* Очередь одна и на физику, и на кадр: submit'ы из двух потоков
     * взаимоисключающие. Fence ждём вне замка. */
    pthread_mutex_t lock;
    /* Замки ОЧЕРЕДЕЙ: Vulkan требует внешней синхронизации VkQueue, но только
     * внутри одной очереди. При двух очередях у каждого потока свой замок —
     * кадр, стоящий в present (FIFO ждёт развёртку), больше не держит такт, а
     * такт, записывающий 200+ диспатчей, не держит кадр. При одной очереди оба
     * берут общий c->lock. */
    pthread_mutex_t qlock[VF_STREAMS];

    char name[256];
    uint32_t api;
};

static uint32_t pick_memory(const vf_ctx *c, uint32_t bits, VkMemoryPropertyFlags want,
                            VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < c->memp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = c->memp.memoryTypes[i].propertyFlags;
        if ((f & want) == want && (f & avoid) == 0) return i;
    }
    return UINT32_MAX;
}

static int has_instance_ext(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *list = calloc(n ? n : 1, sizeof(*list));
    vkEnumerateInstanceExtensionProperties(NULL, &n, list);
    int found = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(list[i].extensionName, name) == 0) { found = 1; break; }
    }
    free(list);
    return found;
}

vf_status vf_init(vf_ctx **out) { return vf_init_ext(out, NULL, 0, 0); }

vf_status vf_init_ext(vf_ctx **out, const char **inst_ext, uint32_t n_ext,
                      int windowed) {
    vf_ctx *c = calloc(1, sizeof(*c));
    if (!c) return ERR(-1, "vmf: нет памяти под контекст");
    pthread_mutex_init(&c->lock, NULL);
    for (uint32_t s = 0; s < VF_STREAMS; s++) pthread_mutex_init(&c->qlock[s], NULL);


    for (uint32_t i = 0; i < n_ext; i++) {
        if (!has_instance_ext(inst_ext[i])) {
            pthread_mutex_destroy(&c->lock);
            free(c);
            return ERR(-9, "vulkan: драйвер не умеет нужное окну расширение");
        }
    }

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "voxelforge";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = n_ext;
    ici.ppEnabledExtensionNames = n_ext ? inst_ext : NULL;
    if (vkCreateInstance(&ici, NULL, &c->instance) != VK_SUCCESS) {
        pthread_mutex_destroy(&c->lock);
        free(c);
        return ERR(-2, "vulkan: не удалось создать instance (нет ICD?)");
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c->instance, &n, NULL);
    if (n == 0) {
        vkDestroyInstance(c->instance, NULL);
        free(c);
        return ERR(-3, "vulkan: ни одного физического устройства");
    }
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    vkEnumeratePhysicalDevices(c->instance, &n, devs);

    /* Семейства очередей: кадру — с графикой (её требует present), такту —
     * compute БЕЗ графики: на RADV это отдельное железное кольцо (4 очереди),
     * и тогда приоритет кадра 1.0 против 0.0 у такта реально решает, кого
     * GPU запускает первым. Если такого семейства нет — такт идёт на семействе
     * кадра второй очередью (если драйвер её даёт), иначе всё на одной. */
    for (uint32_t i = 0; i < n && !c->phys; i++) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
        if (!qn) {
            continue;
        }
        VkQueueFamilyProperties *qf = calloc(qn, sizeof(*qf));
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf);
        int gfx = -1, tick = -1;
        int gfx_is_fallback = 0;
        for (uint32_t q = 0; q < qn; q++) {
            if (gfx < 0 && (qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT))
                gfx = (int)q;
        }
        for (uint32_t q = 0; q < qn; q++) {
            if ((qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfx != (int)q)
                tick = (int)q;
        }
        if (gfx < 0) {
            /* Девайс без графики (например чистый compute): кадровый поток
             * всё равно нужен, он считает кадровые проходы. */
            for (uint32_t q = 0; q < qn; q++) {
                if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    gfx = (int)q;
                    gfx_is_fallback = 1;
                    break;
                }
            }
        }
        if (gfx < 0) {
            free(qf);
            continue;
        }
        c->phys = devs[i];
        c->family = (uint32_t)gfx;
        c->ts_bits = qf[gfx].timestampValidBits;
        c->prio_gfx = 1.0f;
        c->prio_tick = 1.0f;
        c->split = 0;
        /* Две очереди включены ПО УМОЛЧАНИЮ: на RADV у compute своё семейство
         * (приоритет 0.0 против 1.0 у кадра), и кадр перестаёт стоять за
         * тактом. Пользователь запускает `nix run .#live` без переменных, так
         * что новое поведение обязано быть дефолтом, а флаг остаётся на СТАРЫЙ
         * путь для сравнения: VF_QUEUES=one (или VF_SPLIT=0). */
        const char *qenv = getenv("VF_QUEUES");
        const char *sp = qenv ? qenv : getenv("VF_SPLIT");
        int want_split = 1;
        if (sp) want_split = !(strcmp(sp, "0") == 0 || strcmp(sp, "one") == 0);
        if (!want_split) {
            c->tick_family = c->family;
        } else if (!gfx_is_fallback && tick >= 0) {
            c->tick_family = (uint32_t)tick;
            c->ts_bits = qf[tick].timestampValidBits;
            c->prio_tick = 0.0f;
            c->split = 1;
        } else if (!gfx_is_fallback && qf[gfx].queueCount >= 2) {
            /* Одно семейство, но две очереди: приоритеты всё равно разные. */
            c->tick_family = (uint32_t)gfx;
            c->prio_tick = 0.0f;
            c->split = 1;
        } else {
            c->tick_family = (uint32_t)gfx;
        }
        free(qf);
        break;
    }
    free(devs);
    if (!c->phys) {
        vkDestroyInstance(c->instance, NULL);
        pthread_mutex_destroy(&c->lock);
        free(c);
        return ERR(-4, "vulkan: нет устройства с нужной очередью (compute, окну — плюс graphics)");
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(c->phys, &props);
    snprintf(c->name, sizeof(c->name), "%s", props.deviceName);
    c->api = props.apiVersion;
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->memp);

    /* Таймлайн-семафоры: без них два потока не связать «готово до такта N»
     * (бинарный семафор этого не выражает). Фича ядра 1.2 — спрашиваем её
     * явно; если драйвер не умеет, остаёмся на одной очереди, а не пробуем
     * полумеру. */
    VkPhysicalDeviceVulkan12Features have12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 have2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have2.pNext = &have12;
    vkGetPhysicalDeviceFeatures2(c->phys, &have2);
    if (c->split && have12.timelineSemaphore != VK_TRUE) {
        fprintf(stderr, "[queues] драйвер не умеет таймлайн-семафоры: остаёмся на одной очереди\n");
        c->tick_family = c->family;
        c->prio_tick = 1.0f;
        c->split = 0;
    }

    VkDeviceQueueCreateInfo qci[2];
    memset(qci, 0, sizeof(qci));
    float prios_same[2] = {1.0f, 0.0f};
    uint32_t nqci = 0;
    qci[nqci].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci[nqci].queueFamilyIndex = c->family;
    qci[nqci].queueCount = 1;
    qci[nqci].pQueuePriorities = &c->prio_gfx;
    nqci++;
    if (c->split && c->tick_family != c->family) {
        qci[nqci].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci[nqci].queueFamilyIndex = c->tick_family;
        qci[nqci].queueCount = 1;
        qci[nqci].pQueuePriorities = &c->prio_tick;
        nqci++;
    } else if (c->split) {
        qci[0].queueCount = 2;
        qci[0].pQueuePriorities = prios_same;
    }

    /* apiVersion 1.3, но без лишних фич: ядру нужны storage buffers и
     * таймлайн-семафоры. synchronization2 не включаем — барьеры старые. */
    VkPhysicalDeviceVulkan12Features f12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = nqci;
    dci.pQueueCreateInfos = qci;
    if (c->split) dci.pNext = &f12;
    const char *dev_ext[1] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (windowed) {
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_ext;
    }
    if (vkCreateDevice(c->phys, &dci, NULL, &c->dev) != VK_SUCCESS) {
        vkDestroyInstance(c->instance, NULL);
        pthread_mutex_destroy(&c->lock);
        free(c);
        return ERR(-5, "vulkan: не удалось создать логическое устройство");
    }
    vkGetDeviceQueue(c->dev, c->family, 0, &c->queue);
    vkGetDeviceQueue(c->dev, c->tick_family, c->split && c->tick_family == c->family ? 1 : 0,
                     &c->tick_queue);

    /* Очереди: кадровая и такт. Если split = 0, обе ссылки — на одну очередь,
     * и потоки по-прежнему сериализуются замком контекста. */
    fprintf(stderr,
            "[queues] %s · кадр: семейство %u приоритет %.1f · такт: семейство %u приоритет %.1f · %s\n",
            c->name, c->family, (double)c->prio_gfx, c->tick_family, (double)c->prio_tick,
            c->split ? "ДВЕ ОЧЕРЕДИ" : "одна очередь");

    for (uint32_t s = 0; s < VF_STREAMS; s++) {
        uint32_t fam = s == VF_TICK ? c->tick_family : c->family;
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = fam;
        if (vkCreateCommandPool(c->dev, &pci, NULL, &c->q[s].pool) != VK_SUCCESS)
            return ERR(-13, "vulkan: пул команд потока не создался");

        VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = c->q[s].pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = VF_FRAMES;
        vkAllocateCommandBuffers(c->dev, &cbi, c->q[s].cb);

        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (int i = 0; i < VF_FRAMES; i++) vkCreateFence(c->dev, &fci, NULL, &c->q[s].fence[i]);

        VkSemaphoreTypeCreateInfo stci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue = 0;
        VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &stci;
        if (vkCreateSemaphore(c->dev, &sci, NULL, &c->q[s].tl) != VK_SUCCESS)
            return ERR(-14, "vulkan: таймлайн-семафор потока не создался");

        /* Пул дескрипторов СВОЙ на поток: сброс общего пула, пока другой
         * поток ещё исполняет свой command buffer, — это гонка за память
         * дескрипторов (и ровно тот класс ошибок, что даёт SQC/мусорный
         * адрес на RADV). */
        VkDescriptorPoolSize sz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   VF_MAX_SETS * VF_MAX_BINDINGS};
        VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = VF_MAX_SETS;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &sz;
        for (int i = 0; i < VF_FRAMES; i++)
            if (vkCreateDescriptorPool(c->dev, &dpci, NULL, &c->q[s].desc_pool[i]) != VK_SUCCESS)
                return ERR(-6, "vulkan: пул дескрипторов не создался");
    }

    /* Диагностика стадий: пул запросов и host-visible буфер на каждый слот.
     * Всё это — необязательная часть: если драйвер не умеет таймстемпы на
     * compute-очереди, честно помечаем сбор недоступным и живём дальше. */
    c->period_ms = props.limits.timestampPeriod * 1e-6f;
    c->timestamp_cg = props.limits.timestampComputeAndGraphics == VK_TRUE;
    c->perf_ok = c->timestamp_cg && c->ts_bits > 0 && props.limits.timestampPeriod > 0.0f;
    if (c->perf_ok) {
        VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = VF_PERF_MAX;
        for (int i = 0; i < VF_FRAMES; i++) {
            if (vkCreateQueryPool(c->dev, &qpci, NULL, &c->qpool[i]) != VK_SUCCESS ||
                vf_buffer_new(c, (VF_PERF_MAX + 1) * sizeof(uint64_t), 1, &c->tsbuf[i])
                        .code != 0) {
                fprintf(stderr, "[perf] таймстемпы: пул/буфер слота %d не создался\n", i);
                c->perf_ok = 0;
                break;
            }
            c->perf_id[i] = UINT64_MAX;
        }
    }

    *out = c;
    return OK;
}

void vf_shutdown(vf_ctx *c) {
    if (!c) return;
    if (c->dev) vkDeviceWaitIdle(c->dev);
    for (int i = 0; i < VF_FRAMES; i++) {
        if (c->qpool[i] && c->dev) vkDestroyQueryPool(c->dev, c->qpool[i], NULL);
        if (c->tsbuf[i]) vf_buffer_free(c, c->tsbuf[i]);
    }
    for (uint32_t s = 0; s < VF_STREAMS; s++) {
        if (!c->dev) break;
        for (int i = 0; i < VF_FRAMES; i++)
            if (c->q[s].desc_pool[i])
                vkDestroyDescriptorPool(c->dev, c->q[s].desc_pool[i], NULL);
        for (int i = 0; i < VF_FRAMES; i++)
            if (c->q[s].fence[i]) vkDestroyFence(c->dev, c->q[s].fence[i], NULL);
        if (c->q[s].tl) vkDestroySemaphore(c->dev, c->q[s].tl, NULL);
        if (c->q[s].pool) vkDestroyCommandPool(c->dev, c->q[s].pool, NULL);
    }
    if (c->dev) vkDestroyDevice(c->dev, NULL);
    if (c->instance) vkDestroyInstance(c->instance, NULL);
    pthread_mutex_destroy(&c->lock);
    for (uint32_t s = 0; s < VF_STREAMS; s++) pthread_mutex_destroy(&c->qlock[s]);
    free(c);
}

vf_status vf_init_for_window(vf_ctx **out) {
    /* Окно нуждается в VK_KHR_surface и в расширении своей оконной системы.
     * Берём те, что реально есть в драйвере: на чистом Wayland нет xcb. */
    /* Имена строками, а не макросами: макросы оконных расширений требуют
     * VK_USE_PLATFORM_*_KHR и тянут заголовки Wayland/X11 в ядро. */
    static const char *want[] = {
        "VK_KHR_surface",
        "VK_KHR_wayland_surface",
        "VK_KHR_xcb_surface",
        "VK_KHR_xlib_surface",
    };
    const char *ext[4];
    uint32_t n = 0;
    int surface = 0;
    for (int i = 0; i < 4; i++) {
        if (!has_instance_ext(want[i])) continue;
        if (i == 0) surface = 1;
        ext[n++] = want[i];
    }
    if (!surface) return ERR(-10, "vulkan: нет VK_KHR_surface — окно невозможно");
    return vf_init_ext(out, ext, n, 1);
}

vf_status vf_idle(vf_ctx *c) {
    if (!c || !c->dev) return OK;
    if (vkDeviceWaitIdle(c->dev) != VK_SUCCESS) return ERR(-11, "vulkan: устройство не затихло");
    return OK;
}

void *vf_ctx_instance(vf_ctx *c) { return c->instance; }
void *vf_ctx_phys(vf_ctx *c) { return c->phys; }
void *vf_ctx_device(vf_ctx *c) { return c->dev; }
void *vf_ctx_queue(vf_ctx *c) { return c->queue; }
uint32_t vf_ctx_family(vf_ctx *c) { return c->family; }
void vf_ctx_lock(vf_ctx *c) { pthread_mutex_lock(&c->lock); }
void vf_ctx_unlock(vf_ctx *c) { pthread_mutex_unlock(&c->lock); }

/* Замок ОЧЕРЕДИ потока. Vulkan требует внешней синхронизации VkQueue, но не
 * между разными очередями. При двух очередях у такта и кадра свои замки:
 * кадр, стоящий в present (FIFO ждёт развёртку), больше не блокирует submit
 * такта, а долгая запись такта не блокирует кадр. При одной очереди очередь
 * одна — берём общий замок, как раньше. */
void vf_queue_lock(vf_ctx *c, uint32_t stream) {
    if (!c) return;
    if (c->split && stream < VF_STREAMS)
        pthread_mutex_lock(&c->qlock[stream]);
    else
        pthread_mutex_lock(&c->lock);
}
void vf_queue_unlock(vf_ctx *c, uint32_t stream) {
    if (!c) return;
    if (c->split && stream < VF_STREAMS)
        pthread_mutex_unlock(&c->qlock[stream]);
    else
        pthread_mutex_unlock(&c->lock);
}

/* Стадии последнего submit'а потока, мс: запись команд, ожидание замка
 * очереди, vkQueueSubmit, ожидание fence. 0 — submit'а ещё не было. */
void vf_submit_stats(vf_ctx *c, uint32_t stream, float *record_ms, float *lock_ms,
                     float *submit_ms, float *fence_ms) {
    if (!c || stream >= VF_STREAMS) {
        if (record_ms) *record_ms = 0.0f;
        if (lock_ms) *lock_ms = 0.0f;
        if (submit_ms) *submit_ms = 0.0f;
        if (fence_ms) *fence_ms = 0.0f;
        return;
    }
    struct vf_pass *q = &c->q[stream];
    if (record_ms) *record_ms = (float)q->st_record;
    if (lock_ms) *lock_ms = (float)q->st_lock;
    if (submit_ms) *submit_ms = (float)q->st_submit;
    if (fence_ms) *fence_ms = (float)q->st_fence;
}


vf_status vf_device_info(vf_ctx *c, char *name, size_t name_len, uint32_t *api_version) {
    if (!c) return ERR(-1, "нет контекста");
    if (name && name_len) snprintf(name, name_len, "%s", c->name);
    if (api_version) *api_version = c->api;
    return OK;
}

vf_status vf_buffer_new(vf_ctx *c, size_t bytes, int host_visible, vf_buffer **out) {
    if (!c) return ERR(-1, "нет контекста");
    vf_buffer *b = calloc(1, sizeof(*b));
    if (!b) return ERR(-2, "нет памяти под буфер");

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes ? bytes : 4;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    /* Две очереди из РАЗНЫХ семейств: буфер, который читает кадр и пишет
     * такт, обязан быть CONCURRENT, иначе нужны передачи владения
     * (release/acquire) на каждой границе — и пропущенная передача даёт
     * ровно «мусорный адрес» в шейдере. */
    uint32_t fams[2] = {c->family, c->tick_family};
    if (c->split && c->family != c->tick_family) {
        bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bci.queueFamilyIndexCount = 2;
        bci.pQueueFamilyIndices = fams;
    } else {
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    if (vkCreateBuffer(c->dev, &bci, NULL, &b->buf) != VK_SUCCESS) {
        free(b);
        return ERR(-3, "vulkan: буфер не создался");
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->dev, b->buf, &req);

    uint32_t idx;
    if (host_visible) {
        /* host-visible + device-local, если есть (lavapipe и интегрированные
         * GPU дают; дискретные — нет, тогда просто host-visible). */
        idx = pick_memory(c, req.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (idx == UINT32_MAX)
            idx = pick_memory(c, req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    } else {
        idx = pick_memory(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (idx == UINT32_MAX)
            idx = pick_memory(c, req.memoryTypeBits, 0, 0);
    }
    if (idx == UINT32_MAX) {
        vkDestroyBuffer(c->dev, b->buf, NULL);
        free(b);
        return ERR(-4, "vulkan: подходящий тип памяти не найден");
    }

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = idx;
    if (vkAllocateMemory(c->dev, &mai, NULL, &b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(c->dev, b->buf, NULL);
        free(b);
        return ERR(-5, "vulkan: память под буфер не выделилась");
    }
    vkBindBufferMemory(c->dev, b->buf, b->mem, 0);
    if (host_visible) vkMapMemory(c->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->mapped);

    b->bytes = bytes;
    *out = b;
    return OK;
}

void *vf_buffer_handle(vf_buffer *b) { return b ? b->buf : NULL; }

void vf_buffer_free(vf_ctx *c, vf_buffer *b) {
    if (!b) return;
    if (c && c->dev) {
        if (b->mapped) vkUnmapMemory(c->dev, b->mem);
        vkDestroyBuffer(c->dev, b->buf, NULL);
        vkFreeMemory(c->dev, b->mem, NULL);
    }
    free(b);
}

vf_status vf_buffer_write(vf_ctx *c, vf_buffer *b, size_t offset, const void *src, size_t bytes) {
    if (!b->mapped) return ERR(-1, "буфер не host-visible: запись невозможна");
    if (offset + bytes > b->bytes) return ERR(-2, "запись за границей буфера");
    memcpy((char *)b->mapped + offset, src, bytes);
    return OK;
}

vf_status vf_buffer_read(vf_ctx *c, vf_buffer *b, size_t offset, void *dst, size_t bytes) {
    if (!b->mapped) return ERR(-1, "буфер не host-visible: чтение невозможно");
    if (offset + bytes > b->bytes) return ERR(-2, "чтение за границей буфера");
    memcpy(dst, (const char *)b->mapped + offset, bytes);
    return OK;
}

vf_status vf_pipeline_new(vf_ctx *c, const uint32_t *spirv, size_t spirv_bytes,
                          uint32_t n_storage_buffers, uint32_t push_bytes,
                          const char *debug_name, vf_pipeline **out) {
    if (!c) return ERR(-1, "нет контекста");
    if (n_storage_buffers > VF_MAX_BINDINGS) return ERR(-2, "слишком много буферов");

    vf_pipeline *p = calloc(1, sizeof(*p));
    if (!p) return ERR(-3, "нет памяти под пайплайн");
    p->n_buffers = n_storage_buffers;
    p->push_bytes = push_bytes;

    VkDescriptorSetLayoutBinding binds[VF_MAX_BINDINGS];
    memset(binds, 0, sizeof(binds));
    for (uint32_t i = 0; i < n_storage_buffers; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = n_storage_buffers;
    dlci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(c->dev, &dlci, NULL, &p->dsl) != VK_SUCCESS) {
        free(p);
        return ERR(-4, "vulkan: descriptor set layout не создался");
    }

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p->dsl;
    plci.pushConstantRangeCount = push_bytes ? 1 : 0;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(c->dev, &plci, NULL, &p->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
        free(p);
        return ERR(-5, "vulkan: pipeline layout не создался");
    }

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv_bytes;
    smci.pCode = spirv;
    VkShaderModule mod;
    if (vkCreateShaderModule(c->dev, &smci, NULL, &mod) != VK_SUCCESS) {
        vkDestroyPipelineLayout(c->dev, p->layout, NULL);
        vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
        free(p);
        return ERR(-6, "vulkan: SPIR-V не принят (битый или не тот stage?)");
    }

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = p->layout;
    VkResult r = vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipe);
    vkDestroyShaderModule(c->dev, mod, NULL);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(c->dev, p->layout, NULL);
        vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
        free(p);
        return ERR(-7, "vulkan: compute pipeline не создался");
    }

    (void)debug_name;
    *out = p;
    return OK;
}

void vf_pipeline_free(vf_ctx *c, vf_pipeline *p) {
    if (!p) return;
    if (c && c->dev) {
        vkDestroyPipeline(c->dev, p->pipe, NULL);
        vkDestroyPipelineLayout(c->dev, p->layout, NULL);
        vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
    }
    free(p);
}

/* Записать пачку dispatch'ей в слот кольца потока и отправить в его очередь.
 * waits — значения таймлайнов, которых этот submit дожидается ДО начала работы
 * (например «кадр дочитал банк состояния»). Сам submit всегда сигналит
 * таймлайн своего потока значением «номер submit'а + 1»: это и есть «такт N
 * посчитан» без всяких бинарных семафоров. wait_now != 0 — дождаться ЭТОГО
 * submit'а (режим тестов). */
static vf_status submit(vf_ctx *c, vf_dispatch *list, uint32_t count, uint32_t stream,
                        const vf_semwait *waits, uint32_t n_waits, int wait_now) {
    struct vf_pass *q = &c->q[stream];
    uint32_t slot = (uint32_t)(q->submitted % VF_FRAMES);
    if (q->submitted >= VF_FRAMES) {
        /* Слот занят прошлым submit'ом этого потока — ждём его. */
        if (vkWaitForFences(c->dev, 1, &q->fence[slot], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            return ERR(-8, "vulkan: fence слота не дождались");
    }

    vkResetDescriptorPool(c->dev, q->desc_pool[slot], 0);
    vkResetCommandBuffer(q->cb[slot], 0);

    double t_rec = vf_now_ms();
    VkCommandBuffer cb = q->cb[slot];


    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS)
        return ERR(-2, "vulkan: command buffer не начался");

    /* Таймстемпы стадий: count+1 метка (до первого диспатча и после каждого).
     * Метка пишется с ALL_COMMANDS — она честно ждёт окончания предыдущей
     * стадии вместе с барьером, поэтому разность соседних меток и есть время
     * стадии. Копирование в host-visible буфер идёт последней командой: без
     * WAIT_BIT оно не останавливает GPU, а к fence результат уже готов. */
    /* Диагностика стадий — у ТАКТА (именно его таблицу печатает F3). У кадра
     * свой счёт времени: submit → fence на хосте (см. vf_wait_stream). */
    int perf = stream == VF_TICK && c->perf && c->perf_ok;
    if (perf && count + 1 > VF_PERF_MAX) {
        if (!c->perf_warned) {
            fprintf(stderr, "[perf] такт из %u диспатчей > %d: стадии не меряю\n",
                    count, VF_PERF_MAX);
            c->perf_warned = 1;
        }
        perf = 0;
    }
    if (perf) {
        vkCmdResetQueryPool(cb, c->qpool[slot], 0, count + 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            c->qpool[slot], 0);
    }

    for (uint32_t d = 0; d < count; d++) {
        vf_dispatch *dp = &list[d];
        if (!dp->pipeline) return ERR(-3, "dispatch без пайплайна");
        if (dp->n_buffers != dp->pipeline->n_buffers)
            return ERR(-4, "число буферов не совпадает с пайплайном");
        /* Конец предыдущей стадии = начало этой (после её барьера). */
        if (perf && d > 0)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                c->qpool[slot], d);

        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = q->desc_pool[slot];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dp->pipeline->dsl;
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(c->dev, &ai, &set);
        if (r != VK_SUCCESS) return ERR(-5, "дескрипторный пул исчерпан");

        VkDescriptorBufferInfo infos[VF_MAX_BINDINGS];
        VkWriteDescriptorSet writes[VF_MAX_BINDINGS];
        memset(writes, 0, sizeof(writes));
        for (uint32_t i = 0; i < dp->n_buffers; i++) {
            infos[i].buffer = dp->buffers[i]->buf;
            infos[i].offset = 0;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(c->dev, dp->n_buffers, writes, 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dp->pipeline->pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dp->pipeline->layout,
                                0, 1, &set, 0, NULL);
        if (dp->push_bytes)
            vkCmdPushConstants(cb, dp->pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, dp->push_bytes, dp->push);
        vkCmdDispatch(cb, dp->groups[0], dp->groups[1], dp->groups[2]);

        /* Шейдерная запись → шейдерное чтение следующего прохода. */
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }

    /* Последняя метка и копирование результатов. Порядок в буфере: count+1
     * метка, затем по одной на каждый диспатч. */
    if (perf) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            c->qpool[slot], count);
        vkCmdCopyQueryPoolResults(cb, c->qpool[slot], 0, count + 1,
                                  c->tsbuf[slot]->buf, 0, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT);
        c->perf_n[slot] = count;
        for (uint32_t d = 0; d < count; d++) c->perf_stage[slot][d] = list[d].stage;
        /* Индекс этого submit'а: по нему vf_perf_read отличает свежие данные
         * от уже перезаписанных следующим тактом. */
        c->perf_id[slot] = q->submitted;
    } else {
        c->perf_n[slot] = 0;
        c->perf_id[slot] = UINT64_MAX;
    }

    if (vkEndCommandBuffer(cb) != VK_SUCCESS)
        return ERR(-6, "vulkan: command buffer не закрылся");
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    /* Таймлайны: ждём чужие значения (ждать начинает вся очередь, поэтому
     * стадия ALL_COMMANDS), сигналим свой номер. Без двух очередей семафоры
     * не нужны вовсе — тогда submit ровно такой же, как раньше. */
    VkTimelineSemaphoreSubmitInfo tsi = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSemaphore wait_sems[VF_MAX_WAITS];
    VkPipelineStageFlags wait_stages[VF_MAX_WAITS];
    uint64_t wait_vals[VF_MAX_WAITS];
    uint64_t signal_val = q->submitted + 1;
    uint32_t nw = 0;
    if (c->split) {
        nw = n_waits > VF_MAX_WAITS ? VF_MAX_WAITS : n_waits;
        for (uint32_t i = 0; i < nw; i++) {
            wait_sems[i] = c->q[waits[i].stream].tl;
            wait_vals[i] = waits[i].value;
            wait_stages[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        tsi.waitSemaphoreValueCount = nw;
        tsi.pWaitSemaphoreValues = nw ? wait_vals : NULL;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &signal_val;
        si.pNext = &tsi;
        if (nw) {
            si.waitSemaphoreCount = nw;
            si.pWaitSemaphores = wait_sems;
            si.pWaitDstStageMask = wait_stages;
        }
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &q->tl;
    }

    VkQueue vkq = stream == VF_TICK ? c->tick_queue : c->queue;
    /* Submit'ы ОДНОЙ очереди взаимоисключающие: при двух очередях у каждого
     * потока свой замок, при одной — общий (см. vf_queue_lock). Ожидание fence —
     * ВНЕ замка. */
    q->st_record = vf_now_ms() - t_rec;
    double t_lock = vf_now_ms();
    vf_queue_lock(c, stream);
    q->st_lock = vf_now_ms() - t_lock;
    double t_sub = vf_now_ms();
    vkResetFences(c->dev, 1, &q->fence[slot]);
    VkResult sres = vkQueueSubmit(vkq, 1, &si, q->fence[slot]);
    q->st_submit = vf_now_ms() - t_sub;
    vf_queue_unlock(c, stream);
    if (sres != VK_SUCCESS)
        return ERR(-7, "vulkan: submit провалился");
    q->submitted++;
    q->st_fence = 0.0;
    if (wait_now) {
        double t_f = vf_now_ms();
        if (vkWaitForFences(c->dev, 1, &q->fence[slot], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            return ERR(-8, "vulkan: fence не дождались");
        q->st_fence = vf_now_ms() - t_f;
    }
    return OK;
}

vf_status vf_run(vf_ctx *c, vf_dispatch *list, uint32_t count) {
    if (!c) return ERR(-1, "нет контекста");
    if (count == 0) return OK;
    return submit(c, list, count, VF_TICK, NULL, 0, 1);
}

/* Не ждать: результат этого такта читается на следующем (см. vf_wait_back). */
vf_status vf_run_async(vf_ctx *c, vf_dispatch *list, uint32_t count) {
    if (!c) return ERR(-1, "нет контекста");
    if (count == 0) return OK;
    return submit(c, list, count, VF_TICK, NULL, 0, 0);
}

/* Любой поток: 0 — такт (compute, низкий приоритет), 1 — кадр (graphics,
 * приоритет 1.0). waits — на какие таймлайны ждать до старта работы. */
vf_status vf_run_stream(vf_ctx *c, vf_dispatch *list, uint32_t count, uint32_t stream,
                        const vf_semwait *waits, uint32_t n_waits, int wait_now) {
    if (!c) return ERR(-1, "нет контекста");
    if (stream >= VF_STREAMS) return ERR(-2, "нет такого потока submit'ов");
    if (count == 0) return OK;
    return submit(c, list, count, stream, waits, n_waits, wait_now);
}

/* Сколько submit'ов отправлено в поток (= значение его таймлайна). */
uint64_t vf_stream_value(vf_ctx *c, uint32_t stream) {
    if (!c || stream >= VF_STREAMS) return 0;
    return c->q[stream].submitted;
}

/* Дождаться на хосте: поток дошёл до значения value. 0 — дождались, -1 —
 * таймаут. */
int vf_wait_timeline(vf_ctx *c, uint32_t stream, uint64_t value, uint64_t timeout_ns) {
    if (!c || stream >= VF_STREAMS) return -1;
    if (value == 0) return 0;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &c->q[stream].tl;
    wi.pValues = &value;
    VkResult r = vkWaitSemaphores(c->dev, &wi, timeout_ns ? timeout_ns : UINT64_MAX);
    return r == VK_SUCCESS ? 0 : -1;
}

int vf_split(vf_ctx *c) { return c ? c->split : 0; }

void vf_queue_info(vf_ctx *c, uint32_t *gfx_family, uint32_t *tick_family, float *prio_gfx,
                   float *prio_tick) {
    if (!c) return;
    if (gfx_family) *gfx_family = c->family;
    if (tick_family) *tick_family = c->tick_family;
    if (prio_gfx) *prio_gfx = c->prio_gfx;
    if (prio_tick) *prio_tick = c->prio_tick;
}

/* Очередь кадра (graphics) — её берёт окно: present обязан поддерживаться
 * именно этим семейством. */
void *vf_ctx_queue_tick(vf_ctx *c) { return c ? c->tick_queue : NULL; }

/* Таймлайн-семафор потока наружу: окно сигналит кадровый семафор своим
 * submit'ом и ждёт тактовый, не заводя собственных. */
void *vf_ctx_semaphore(vf_ctx *c, uint32_t stream) {
    if (!c || stream >= VF_STREAMS) return NULL;
    return c->q[stream].tl;
}

/* Банк состояния тел: «кадр N прочитал банк b». Такт перед записью в этот банк
 * ждёт это значение — иначе он перезапишет то, что кадр ещё читает. */
void vf_note_bank_read(vf_ctx *c, uint32_t bank, uint64_t frame_value) {
    if (!c || bank >= VF_BANKS) return;
    pthread_mutex_lock(&c->lock);
    if (frame_value > c->bank_read[bank]) c->bank_read[bank] = frame_value;
    pthread_mutex_unlock(&c->lock);
}

uint64_t vf_bank_read_value(vf_ctx *c, uint32_t bank) {
    if (!c || bank >= VF_BANKS) return 0;
    return c->bank_read[bank];
}

/* Дождаться submit'а потока, отправленного `back` назад (0 — последний). */
vf_status vf_wait_stream(vf_ctx *c, uint32_t stream, uint32_t back) {
    if (!c || stream >= VF_STREAMS) return OK;
    struct vf_pass *q = &c->q[stream];
    if (q->submitted == 0) return OK;
    if (back + 1 > q->submitted) back = (uint32_t)(q->submitted - 1);
    uint32_t slot = (uint32_t)((q->submitted - 1 - back) % VF_FRAMES);
    if (vkWaitForFences(c->dev, 1, &q->fence[slot], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return ERR(-8, "vulkan: fence не дождались");
    return OK;
}

/* Дождаться такта, отправленного `back` submit'ов назад (1 — предыдущий).
 * К моменту, когда физика просит статистику (такт раз в 50 мс), он обычно
 * уже закрыт, и ожидание стоит ноль. */
vf_status vf_wait_back(vf_ctx *c, uint32_t back) { return vf_wait_stream(c, VF_TICK, back); }

/* ---------------------------------------------------------------- диагностика */

int vf_perf_supported(vf_ctx *c) { return c && c->perf_ok; }

int vf_perf_on(vf_ctx *c) { return c && c->perf && c->perf_ok; }

vf_status vf_perf_enable(vf_ctx *c, int on) {
    if (!c) return ERR(-1, "нет контекста");
    if (!on) {
        c->perf = 0;
        fprintf(stderr, "[perf] сбор стадий выключен\n");
        return OK;
    }
    if (!c->perf_ok) {
        fprintf(stderr,
                "[perf] GPU-таймстемпы недоступны: timestampComputeAndGraphics=%d, "
                "timestampValidBits=%u, timestampPeriod=%.3f нс — сбор выключен\n",
                c->timestamp_cg, c->ts_bits, (double)c->period_ms * 1e6);
        return ERR(-12, "таймстемпы не поддерживаются на этой очереди");
    }
    c->perf = 1;
    fprintf(stderr,
            "[perf] GPU-таймстемпы включены: %s · период %.3f нс · validBits %u · такт до %d стадий\n",
            c->name, (double)c->period_ms * 1e6, c->ts_bits, VF_PERF_MAX);
    return OK;
}

int vf_perf_read(vf_ctx *c, uint32_t back, float *ms, uint32_t *stage, int max,
                 float *tick_ms) {
    if (!c || !c->perf || !c->perf_ok || !ms || !stage || max <= 0) return -1;
    uint64_t n_sub = c->q[VF_TICK].submitted;
    if (n_sub == 0 || back + 1 > n_sub) return -1;
    uint64_t id = n_sub - 1 - back;
    uint32_t slot = (uint32_t)(id % VF_FRAMES);
    /* Слот успел уйти под следующий такт — данные этого уже потеряны. */
    if (c->perf_id[slot] != id || c->perf_n[slot] == 0) return -1;

    uint32_t n = c->perf_n[slot];
    uint64_t ts[VF_PERF_MAX + 1];
    if (vf_buffer_read(c, c->tsbuf[slot], 0, ts, (size_t)(n + 1) * sizeof(uint64_t)).code != 0)
        return -1;

    uint32_t out = n < (uint32_t)max ? n : (uint32_t)max;
    uint64_t total = 0;
    for (uint32_t k = 0; k < out; k++) {
        /* Обратный ход метки — не «минус время», а мусор: 0 честнее. */
        uint64_t d = ts[k + 1] >= ts[k] ? ts[k + 1] - ts[k] : 0;
        total += d;
        ms[k] = (float)d * c->period_ms;
        stage[k] = c->perf_stage[slot][k];
    }
    /* Время такта — СУММА соседних разностей, а не ts[n] - ts[0]: у драйверов
     * с 32 значащими битами метка за такт может переполниться, и прямая
     * разность даёт ноль, а сумма дельт остаётся верной. */
    if (tick_ms) *tick_ms = (float)total * c->period_ms;
    return (int)out;
}
