// Живое окно движка: SDL3 + Vulkan. Логики мира здесь нет — только
// поверхность, swapchain, два конвейера (меши и линии) и ввод.
#include "voxelforge_view.h"
#include "vf_font5x7.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SHAPES 512
#define MAX_SWAP 8
/* Оверлей стадий: не больше стольких строк и столько байт под его вершины.
 * Потолок держим с запасом: строка текста в 18 знаков при scale 2 — это
 * ~600 вершин, 16 строк — ~10 тыс. вершин, 240 КБ. Буфер 2 МБ заведомо выше. */
#define VF_PERF_MAX_ROWS 24
/* История частот для графиков: 120 кадров ≈ 2 с при 60 fps. */
#define VF_HIST_MAX 120
#define VF_UI_VBUF_BYTES (2u * 1024u * 1024u)

typedef struct gpu_shape {
    VkBuffer vbuf, ibuf;
    VkDeviceMemory vmem, imem;
    uint32_t n_idx;
    int is_lines;
} gpu_shape;

struct vf_view {
    vf_ctx *core;   /* девайс ядра — единственный на всё */
    SDL_Window *win;
    VkInstance inst;
    VkSurfaceKHR surf;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    char gpu_name[256];

    VkSwapchainKHR sc;
    uint32_t sc_n;
    VkImage sc_img[MAX_SWAP];
    VkImageView sc_view[MAX_SWAP];
    VkFormat sc_fmt;
    int present_mode; /* выбранный VkPresentModeKHR — окну и приложению */
    VkExtent2D extent;

    VkImage depth;
    VkDeviceMemory depth_mem;
    VkImageView depth_view;

    VkRenderPass rp;
    VkFramebuffer fb[MAX_SWAP];
    int fb_count;

    VkPipeline pipe_mesh, pipe_line;
    VkPipelineLayout lay_mesh, lay_line;
    /* Оверлей диагностики: треугольники в NDC с цветом на вершине, без
     * дескрипторов и push-констант. Вершины собираются на CPU каждый кадр. */
    VkPipeline pipe_ui;
    VkPipelineLayout lay_ui;
    VkBuffer ui_vbuf;
    VkDeviceMemory ui_vmem;
    float *ui_ptr;
    uint32_t ui_n;      /* вершин в буфере за этот кадр */
    uint32_t ui_cap;    /* вершин максимум */
    vf_perf_row perf_rows[VF_PERF_MAX_ROWS];
    uint32_t perf_n;
    /* ИТОГО такта за то же окно: мин/сред/макс, мс. */
    float perf_tick_min_ms, perf_tick_mean_ms, perf_tick_max_ms;
    /* Строка тел/контактов под таблицей; scene_ok == 0 — не рисовать. */
    vf_scene_counts perf_scene;
    int perf_scene_ok;
    /* Оверлей запрошен этим кадром (F3 / VF_PERF), даже если данных ещё нет:
     * тогда панель показывает «ждём такт», а не исчезает молча. */
    int perf_show;
    /* Частоты кадра и физики: среднее и крайние за окно + история на графики.
     * Копия vf_rates внутри окна — указатели Rust-истории живут один вызов. */
    float fps_mean, fps_min, fps_max;
    float tps_mean, tps_min, tps_max;
    float frame_mean_ms;
    uint32_t rates_frames;
    float hist_fps[VF_HIST_MAX];
    float hist_tps[VF_HIST_MAX];
    uint32_t hist_n;
    int rates_ok;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    VkBuffer body_buf;   /* буфер тел физики (тот же девайс) */
    uint32_t body_n;
    VkBuffer ubo;
    VkDeviceMemory ubo_mem;
    void *ubo_ptr;

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkSemaphore sem_ready, sem_done;
    VkFence fence;
    int recording;
    uint32_t cur_img;

    gpu_shape shapes[MAX_SHAPES];
    int n_shapes;
    /* Прошлое состояние F3 для фронта: событие SDL приходит и с автоповтором, а
     * опрос состояния — только «зажата», поэтому переключение ловим по фронту
     * 0→1, а не по самому факту удержания. */
    int f3_prev;

    float vp[16];
    float eye[4];
    float far_m;
    const char *shader_dir;
};

static int ok(VkResult r, const char *what) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[view] %s: VkResult %d\n", what, (int)r);
        return 0;
    }
    return 1;
}

static vf_status st(int32_t code, const char *msg) {
    vf_status s = {code, msg};
    return s;
}

static VkShaderModule load_shader(VkDevice dev, const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[view] нет шейдера %s\n", path);
        return VK_NULL_HANDLE;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *code = malloc((size_t)n);
    if (fread(code, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(code);
        return VK_NULL_HANDLE;
    }
    fclose(f);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = (size_t)n;
    ci.pCode = (const uint32_t *)code;
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(dev, &ci, NULL, &m);
    free(code);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[view] vkCreateShaderModule(%s) = %d\n", name, (int)r);
        return VK_NULL_HANDLE;
    }
    return m;
}

static uint32_t mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return 0;
}

static int make_buffer(vf_view *v, VkDeviceSize size, VkBufferUsageFlags usage,
                       VkBuffer *buf, VkDeviceMemory *mem, VkMemoryPropertyFlags props) {
    /* Проверки перед драйвером: на нулевом размере и при исчерпанной памяти
     * RADV падал внутри vkBindBufferMemory вместо того, чтобы вернуть ошибку. */
    if (size == 0) {
        fprintf(stderr, "[view] буфер нулевого размера — пропуск\n");
        return 0;
    }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!ok(vkCreateBuffer(v->dev, &bi, NULL, buf), "vkCreateBuffer")) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(v->dev, *buf, &req);
    uint32_t mti = mem_type(v->phys, req.memoryTypeBits, props);
    if (mti == UINT32_MAX) {
        fprintf(stderr, "[view] нет подходящего типа памяти под буфер %.1f МБ\n",
                (double)req.size / (1024.0 * 1024.0));
        vkDestroyBuffer(v->dev, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return 0;
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mti;
    if (!ok(vkAllocateMemory(v->dev, &ai, NULL, mem), "vkAllocateMemory")) {
        vkDestroyBuffer(v->dev, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return 0;
    }
    if (!ok(vkBindBufferMemory(v->dev, *buf, *mem, 0), "vkBindBufferMemory")) {
        vkDestroyBuffer(v->dev, *buf, NULL);
        vkFreeMemory(v->dev, *mem, NULL);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

static void free_buffer(vf_view *v, VkBuffer buf, VkDeviceMemory mem) {
    if (buf) vkDestroyBuffer(v->dev, buf, NULL);
    if (mem) vkFreeMemory(v->dev, mem, NULL);
}

// ---------------------------------------------------------------- инициализация


static int make_swapchain(vf_view *v);
static int make_swapchain_shaders(vf_view *v);

static int make_depth(vf_view *v) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_D32_SFLOAT;
    ci.extent.width = v->extent.width;
    ci.extent.height = v->extent.height;
    ci.extent.depth = 1;
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!ok(vkCreateImage(v->dev, &ci, NULL, &v->depth), "vkCreateImage(depth)")) return 0;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(v->dev, v->depth, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mem_type(v->phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!ok(vkAllocateMemory(v->dev, &ai, NULL, &v->depth_mem), "alloc depth")) return 0;
    vkBindImageMemory(v->dev, v->depth, v->depth_mem, 0);
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = v->depth;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_D32_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    return ok(vkCreateImageView(v->dev, &vi, NULL, &v->depth_view), "depth view");
}

static int make_render_pass(vf_view *v) {
    VkAttachmentDescription att[2] = {0};
    att[0].format = v->sc_fmt;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    att[1].format = VK_FORMAT_D32_SFLOAT;
    att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference col = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dep = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &col;
    sub.pDepthStencilAttachment = &dep;

    VkSubpassDependency d[2] = {0};
    d[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    d[0].dstSubpass = 0;
    d[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    d[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    d[0].srcAccessMask = 0;
    d[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    d[1] = d[0];
    d[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    d[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    d[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 2;
    ci.pAttachments = att;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 2;
    ci.pDependencies = d;
    return ok(vkCreateRenderPass(v->dev, &ci, NULL, &v->rp), "render pass");
}

static int make_framebuffers(vf_view *v) {
    v->fb_count = 0;
    for (uint32_t i = 0; i < v->sc_n; i++) {
        VkImageView att[2] = {v->sc_view[i], v->depth_view};
        VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = v->rp;
        ci.attachmentCount = 2;
        ci.pAttachments = att;
        ci.width = v->extent.width;
        ci.height = v->extent.height;
        ci.layers = 1;
        if (!ok(vkCreateFramebuffer(v->dev, &ci, NULL, &v->fb[i]), "framebuffer")) return 0;
        v->fb_count++;
    }
    return 1;
}

static VkPipelineLayout make_layout(vf_view *v) {
    VkPushConstantRange pc = {0};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.size = 80; // mat4 + vec4
    VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &v->dsl;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pc;
    VkPipelineLayout lay = VK_NULL_HANDLE;
    if (!ok(vkCreatePipelineLayout(v->dev, &ci, NULL, &lay), "pipeline layout")) return VK_NULL_HANDLE;
    return lay;
}

static VkPipeline make_pipeline(vf_view *v, VkPipelineLayout lay, VkShaderModule vs,
                                VkShaderModule fs, int lines) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";

    VkVertexInputBindingDescription bind = {0, sizeof(float) * (lines ? 3 : 6), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                                 {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3}};
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &bind;
    vin.vertexAttributeDescriptionCount = lines ? 1 : 2;
    vin.pVertexAttributeDescriptions = attr;

    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = lines ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = lines ? VK_FALSE : VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dv = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dv.dynamicStateCount = 2;
    dv.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = st;
    ci.pVertexInputState = &vin;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dv;
    ci.layout = lay;
    ci.renderPass = v->rp;
    VkPipeline p = VK_NULL_HANDLE;
    if (!ok(vkCreateGraphicsPipelines(v->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p), "graphics pipeline"))
        return VK_NULL_HANDLE;
    return p;
}

/* Конвейер оверлея: NDC-треугольники с цветом на вершине, без дескрипторов,
 * без глубины, с альфа-смешением (панель полупрозрачная). */
static VkPipeline make_ui_pipeline(vf_view *v, VkPipelineLayout lay, VkShaderModule vs,
                                   VkShaderModule fs) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";

    VkVertexInputBindingDescription bind = {0, sizeof(float) * 6, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr[2] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 2},
    };
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &bind;
    vin.vertexAttributeDescriptionCount = 2;
    vin.pVertexAttributeDescriptions = attr;

    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cba = {0};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dv = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dv.dynamicStateCount = 2;
    dv.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = st;
    ci.pVertexInputState = &vin;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dv;
    ci.layout = lay;
    ci.renderPass = v->rp;
    VkPipeline p = VK_NULL_HANDLE;
    if (!ok(vkCreateGraphicsPipelines(v->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p), "ui pipeline"))
        return VK_NULL_HANDLE;
    return p;
}

static int make_swapchain(vf_view *v) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(v->phys, v->surf, &caps);

    uint32_t fmt_n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(v->phys, v->surf, &fmt_n, NULL);
    VkSurfaceFormatKHR fmts[MAX_SWAP * 4];
    if (fmt_n > MAX_SWAP * 4) fmt_n = MAX_SWAP * 4;
    vkGetPhysicalDeviceSurfaceFormatsKHR(v->phys, v->surf, &fmt_n, fmts);
    VkSurfaceFormatKHR pick = fmts[0];
    for (uint32_t i = 0; i < fmt_n; i++) {
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
            fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            pick = fmts[i];
            break;
        }
    }
    v->sc_fmt = pick.format;

    if (caps.currentExtent.width != 0xFFFFFFFF) {
        v->extent = caps.currentExtent;
    } else {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(v->win, &w, &h);
        v->extent.width = (uint32_t)w;
        v->extent.height = (uint32_t)h;
    }

    /* Минимальное число образов, которое даёт драйвер. Меньше нельзя —
     * Vulkan требует imageCount >= minImageCount. Один образ не создастся,
     * поэтому «не копить кадры» решается режимом показа, а не числом образов. */
    /* Образов ровно минимум, который даёт драйвер: с FIFO лишние образы
     * позволяют циклу убегать вперёд, и показ начинает выкидывать кадры
     * неравномерно — объекты идут рывками. На минимуме acquire блокирует, и
     * приложение жёстко привязано к развёртке. */
    uint32_t want = caps.minImageCount;
    if (getenv("VF_SWAP")) {
        uint32_t n = (uint32_t)atoi(getenv("VF_SWAP"));
        if (n >= caps.minImageCount) want = n;
    }
    if (caps.maxImageCount > 0 && want > caps.maxImageCount) want = caps.maxImageCount;

    /* Режим показа: MAILBOX — очередь не копится, показывается самый свежий
     * кадр (без разрывов); IMMEDIATE — вообще без ожидания вертикальной
     * синхронизации. FIFO оставляем последним: он как раз и держит очередь в
     * N кадров, из-за чего картинка отстаёт от физики. */
    uint32_t pm_n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(v->phys, v->surf, &pm_n, NULL);
    VkPresentModeKHR modes[16];
    if (pm_n > 16) pm_n = 16;
    if (pm_n) vkGetPhysicalDeviceSurfacePresentModesKHR(v->phys, v->surf, &pm_n, modes);
    int has_mailbox = 0, has_immediate = 0;
    for (uint32_t i = 0; i < pm_n; i++) {
        if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) has_mailbox = 1;
        if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) has_immediate = 1;
    }
    /* По умолчанию FIFO: показ привязан к развёртке, и движение камеры идёт
     * ровно. MAILBOX отдаёт ритм приложению — с ним картинка отзывчивее, но
     * фаза у композитора не та, и на движении видно подрагивание.
     * VF_PRESENT=mailbox|immediate включает их осознанно. */
    /* По умолчанию FIFO: показ привязан к развёртке, и движение камеры идёт
     * ровно. MAILBOX доступен через VF_PRESENT=mailbox (с ним нет копления
     * очереди, но движение мельче дёргается — его не берём за дефолт). */
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    const char *pm_name = "FIFO";
    const char *want_pm = getenv("VF_PRESENT");
    if (want_pm && !strcmp(want_pm, "immediate") && has_immediate) {
        pm = VK_PRESENT_MODE_IMMEDIATE_KHR;
        pm_name = "IMMEDIATE";
    } else if (want_pm && !strcmp(want_pm, "mailbox") && has_mailbox) {
        pm = VK_PRESENT_MODE_MAILBOX_KHR;
        pm_name = "MAILBOX";
    }
    v->present_mode = (int)pm;

    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = v->surf;
    ci.minImageCount = want;
    ci.imageFormat = pick.format;
    ci.imageColorSpace = pick.colorSpace;
    ci.imageExtent = v->extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;
    if (!ok(vkCreateSwapchainKHR(v->dev, &ci, NULL, &v->sc), "swapchain")) return 0;
    fprintf(stderr, "[view] swapchain: %u образов, режим %s\n", want, pm_name);

    v->sc_n = 0;
    vkGetSwapchainImagesKHR(v->dev, v->sc, &v->sc_n, NULL);
    if (v->sc_n > MAX_SWAP) v->sc_n = MAX_SWAP;
    vkGetSwapchainImagesKHR(v->dev, v->sc, &v->sc_n, v->sc_img);
    for (uint32_t i = 0; i < v->sc_n; i++) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = v->sc_img[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = v->sc_fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (!ok(vkCreateImageView(v->dev, &vi, NULL, &v->sc_view[i]), "swapchain view")) return 0;
    }
    return 1;
}

static void drop_swapchain(vf_view *v) {
    for (uint32_t i = 0; i < v->fb_count; i++) vkDestroyFramebuffer(v->dev, v->fb[i], NULL);
    v->fb_count = 0;
    if (v->depth_view) vkDestroyImageView(v->dev, v->depth_view, NULL);
    if (v->depth) vkDestroyImage(v->dev, v->depth, NULL);
    if (v->depth_mem) vkFreeMemory(v->dev, v->depth_mem, NULL);
    v->depth_view = VK_NULL_HANDLE;
    v->depth = VK_NULL_HANDLE;
    v->depth_mem = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < v->sc_n; i++) vkDestroyImageView(v->dev, v->sc_view[i], NULL);
    v->sc_n = 0;
    if (v->sc) vkDestroySwapchainKHR(v->dev, v->sc, NULL);
    v->sc = VK_NULL_HANDLE;
}

vf_status vf_view_new(vf_view **out, vf_ctx *core, int w, int h, const char *title,
                      const char *shader_dir) {
    *out = NULL;
    if (!core) return st(1, "окно без девайса ядра");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "[view] SDL_Init: %s\n", SDL_GetError());
        return st(1, "SDL_Init failed");
    }
    vf_view *v = calloc(1, sizeof(vf_view));
    v->core = core;
    v->shader_dir = shader_dir;
    /* Девайс, очередь и семейство — ядра. Ядро уже создало instance с
     * расширениями окна (vf_init_ext), поэтому surface делается на нём же:
     * буферы тел, контактов и геометрии видны и compute, и графике. */
    v->inst = (VkInstance)vf_ctx_instance(core);
    v->phys = (VkPhysicalDevice)vf_ctx_phys(core);
    v->dev = (VkDevice)vf_ctx_device(core);
    v->queue = (VkQueue)vf_ctx_queue(core);
    v->qfam = vf_ctx_family(core);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(v->phys, &props);
    snprintf(v->gpu_name, sizeof(v->gpu_name), "%s", props.deviceName);

    v->win = SDL_CreateWindow(title, w, h, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!v->win) {
        fprintf(stderr, "[view] SDL_CreateWindow: %s\n", SDL_GetError());
        free(v);
        return st(2, "SDL_CreateWindow failed");
    }
    if (!SDL_Vulkan_CreateSurface(v->win, v->inst, NULL, &v->surf)) {
        fprintf(stderr, "[view] SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
        return st(4, "surface failed");
    }

    /* Одна очередь на физику и на кадр: present обязан поддерживаться именно
     * семейством ядра, иначе окно пришлось бы вешать на другой девайс и весь
     * смысл общего девайса пропадал. */
    VkBool32 sup = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(v->phys, v->qfam, v->surf, &sup);
    if (!sup) {
        fprintf(stderr, "[view] семейство очереди %u не умеет present\n", v->qfam);
        return st(5, "queue family cannot present");
    }

    if (!make_swapchain(v)) return st(7, "swapchain");

    // дескриптор кадра: 0 — UBO кадра, 1 — буфер тел физики (шаг 1: поза
    // приходит на устройство, а не матрицей из CPU)
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 2;
    dl.pBindings = b;
    if (!ok(vkCreateDescriptorSetLayout(v->dev, &dl, NULL, &v->dsl), "dsl")) return st(8, "dsl");

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
    };
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    if (!ok(vkCreateDescriptorPool(v->dev, &dpi, NULL, &v->dpool), "dpool")) return st(9, "dpool");

    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = v->dpool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &v->dsl;
    if (!ok(vkAllocateDescriptorSets(v->dev, &da, &v->dset), "dset")) return st(10, "dset");

    if (!make_buffer(v, sizeof(float) * 24, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &v->ubo, &v->ubo_mem,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return st(11, "ubo");
    vkMapMemory(v->dev, v->ubo_mem, 0, VK_WHOLE_SIZE, 0, &v->ubo_ptr);
    VkDescriptorBufferInfo dbi = {v->ubo, 0, sizeof(float) * 24};
    VkWriteDescriptorSet wd = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet = v->dset;
    wd.dstBinding = 0;
    wd.descriptorCount = 1;
    wd.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wd.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(v->dev, 1, &wd, 0, NULL);

    /* Вершинный буфер оверлея: host-visible и постоянно замапен — вершины
     * собираются на CPU каждый кадр, а кадр начинается после fence, так что
     * прошлый кадр этот буфер уже не читает. */
    if (!make_buffer(v, VF_UI_VBUF_BYTES, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &v->ui_vbuf,
                     &v->ui_vmem,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return st(18, "ui vbuf");
    vkMapMemory(v->dev, v->ui_vmem, 0, VK_WHOLE_SIZE, 0, (void **)&v->ui_ptr);
    v->ui_cap = VF_UI_VBUF_BYTES / (sizeof(float) * 6);

    // Порядок важен: конвейерам нужен render pass, а ему — формат swapchain.
    if (!make_depth(v)) return st(13, "depth");
    if (!make_render_pass(v)) return st(14, "render pass");
    if (!make_swapchain_shaders(v)) return st(12, "pipelines");
    if (!make_framebuffers(v)) return st(15, "framebuffers");

    VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = v->qfam;
    if (!ok(vkCreateCommandPool(v->dev, &cp, NULL, &v->cmd_pool), "cmd pool")) return st(16, "cmd pool");
    VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = v->cmd_pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    if (!ok(vkAllocateCommandBuffers(v->dev, &ca, &v->cmd), "cmd buffer")) return st(17, "cmd");

    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(v->dev, &si, NULL, &v->sem_ready);
    vkCreateSemaphore(v->dev, &si, NULL, &v->sem_done);
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(v->dev, &fi, NULL, &v->fence);

    *out = v;
    fprintf(stderr, "[view] %s · %ux%u\n", v->gpu_name, v->extent.width, v->extent.height);
    return st(0, NULL);
}

// Вынесено вперёд: собирает оба конвейера из SPIR-V.
static int make_swapchain_shaders(vf_view *v) {
    VkShaderModule vs = load_shader(v->dev, v->shader_dir, "view_mesh.vert.spv");
    VkShaderModule fs = load_shader(v->dev, v->shader_dir, "view_mesh.frag.spv");
    VkShaderModule lvs = load_shader(v->dev, v->shader_dir, "view_line.vert.spv");
    VkShaderModule lfs = load_shader(v->dev, v->shader_dir, "view_line.frag.spv");
    if (!vs || !fs || !lvs || !lfs) return 0;
    v->lay_mesh = make_layout(v);
    v->lay_line = make_layout(v);
    if (!v->lay_mesh || !v->lay_line) return 0;
    v->pipe_mesh = make_pipeline(v, v->lay_mesh, vs, fs, 0);
    v->pipe_line = make_pipeline(v, v->lay_line, lvs, lfs, 1);
    vkDestroyShaderModule(v->dev, vs, NULL);
    vkDestroyShaderModule(v->dev, fs, NULL);
    vkDestroyShaderModule(v->dev, lvs, NULL);
    vkDestroyShaderModule(v->dev, lfs, NULL);
    if (!v->pipe_mesh || !v->pipe_line) return 0;

    /* Оверлей диагностики — необязателен: если SPIR-V для него нет, окно
     * обязано подняться, а оверлей просто не рисуется (числа остаются в
     * консоли). Поэтому ошибки здесь не валят инициализацию. */
    VkShaderModule uvs = load_shader(v->dev, v->shader_dir, "view_ui.vert.spv");
    VkShaderModule ufs = load_shader(v->dev, v->shader_dir, "view_ui.frag.spv");
    if (uvs && ufs) {
        VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (ok(vkCreatePipelineLayout(v->dev, &lci, NULL, &v->lay_ui), "pipeline layout(ui)"))
            v->pipe_ui = make_ui_pipeline(v, v->lay_ui, uvs, ufs);
    }
    if (uvs) vkDestroyShaderModule(v->dev, uvs, NULL);
    if (ufs) vkDestroyShaderModule(v->dev, ufs, NULL);
    if (!v->pipe_ui) fprintf(stderr, "[view] оверлей стадий недоступен (нет view_ui SPIR-V)\n");
    return 1;
}

/* Режим показа: 0 — FIFO (развёртка сама задаёт ритм, ограничитель кадра
 * не нужен), иначе MAILBOX/IMMEDIATE — там ритм задаёт приложение. */
int vf_view_present_mode(vf_view *v) {
    if (!v) return 0;
    return v->present_mode == VK_PRESENT_MODE_FIFO_KHR ? 0 : 1;
}

void vf_view_free(vf_view *v) {
    if (!v) return;
    vkDeviceWaitIdle(v->dev);
    for (int i = 0; i < v->n_shapes; i++) {
        free_buffer(v, v->shapes[i].vbuf, v->shapes[i].vmem);
        if (!v->shapes[i].is_lines) free_buffer(v, v->shapes[i].ibuf, v->shapes[i].imem);
    }
    drop_swapchain(v);
    if (v->pipe_mesh) vkDestroyPipeline(v->dev, v->pipe_mesh, NULL);
    if (v->pipe_line) vkDestroyPipeline(v->dev, v->pipe_line, NULL);
    if (v->pipe_ui) vkDestroyPipeline(v->dev, v->pipe_ui, NULL);
    if (v->lay_mesh) vkDestroyPipelineLayout(v->dev, v->lay_mesh, NULL);
    if (v->lay_line) vkDestroyPipelineLayout(v->dev, v->lay_line, NULL);
    if (v->lay_ui) vkDestroyPipelineLayout(v->dev, v->lay_ui, NULL);
    if (v->rp) vkDestroyRenderPass(v->dev, v->rp, NULL);
    if (v->dpool) vkDestroyDescriptorPool(v->dev, v->dpool, NULL);
    if (v->dsl) vkDestroyDescriptorSetLayout(v->dev, v->dsl, NULL);
    free_buffer(v, v->ubo, v->ubo_mem);
    free_buffer(v, v->ui_vbuf, v->ui_vmem);
    if (v->sem_ready) vkDestroySemaphore(v->dev, v->sem_ready, NULL);
    if (v->sem_done) vkDestroySemaphore(v->dev, v->sem_done, NULL);
    if (v->fence) vkDestroyFence(v->dev, v->fence, NULL);
    if (v->cmd_pool) vkDestroyCommandPool(v->dev, v->cmd_pool, NULL);
    /* девайс и instance принадлежат ядру — здесь только своё */
    if (v->surf) vkDestroySurfaceKHR(v->inst, v->surf, NULL);
    if (v->win) SDL_DestroyWindow(v->win);
    SDL_Quit();
    free(v);
}

// ---------------------------------------------------------------- ввод

int vf_view_poll(vf_view *v, vf_input *in) {
    memset(in, 0, sizeof(*in));
    in->alive = 1;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:
            in->alive = 0;
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED:
            in->resized = 1;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            in->dx += e.motion.xrel;
            in->dy += e.motion.yrel;
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            in->wheel += e.wheel.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (e.button.button == SDL_BUTTON_LEFT) in->lmb_pressed = 1;
            break;
        case SDL_EVENT_KEY_DOWN:
            switch (e.key.key) {
            case SDLK_SPACE: in->key_space = 1; break;
            case SDLK_R: in->key_r = 1; break;
            case SDLK_T: in->key_t = 1; break;
            case SDLK_F: in->key_f = 1; break;
            /* F3 — переключатель диагностики стадий (как в Minecraft). Автоповтор
             * SDL не считаем: иначе удержание клавиши мигало бы таблицей. */
            case SDLK_F3:
                if (!e.key.repeat) in->key_f3 = 1;
                break;
            case SDLK_ESCAPE: in->alive = 0; break;
            default: break;
            }
            /* Раскладка может дать другой keycode на той же физической клавише —
             * F3 ловим ещё и по scancode. */
            if (e.key.scancode == SDL_SCANCODE_F3 && !e.key.repeat) in->key_f3 = 1;
            break;
        default:
            break;
        }
    }
    // Удержание клавиш: события дают только нажатие, а полёту нужен
    // непрерывный ввод, пока клавиша зажата.
    const bool *ks = SDL_GetKeyboardState(NULL);
    if (ks) {
        in->key_w = ks[SDL_SCANCODE_W];
        in->key_a = ks[SDL_SCANCODE_A];
        in->key_s = ks[SDL_SCANCODE_S];
        in->key_d = ks[SDL_SCANCODE_D];
        in->key_q = ks[SDL_SCANCODE_Q];
        in->key_e = ks[SDL_SCANCODE_E];
        in->shift = ks[SDL_SCANCODE_LSHIFT] || ks[SDL_SCANCODE_RSHIFT];
        in->ctrl = ks[SDL_SCANCODE_LCTRL] || ks[SDL_SCANCODE_RCTRL];
        /* F3 — переключатель, поэтому важен только фронт нажатия: удержание
         * (и автоповтор) не должны мигать диагностикой. */
        int f3 = ks[SDL_SCANCODE_F3] ? 1 : 0;
        if (f3 && !v->f3_prev) in->key_f3 = 1;
        v->f3_prev = f3;
    }
    SDL_MouseButtonFlags flags = SDL_GetMouseState(&in->mx, &in->my);
    in->lmb = (flags & SDL_BUTTON_LMASK) != 0;
    in->rmb = (flags & SDL_BUTTON_RMASK) != 0;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(v->win, &w, &h);
    in->width = (uint32_t)w;
    in->height = (uint32_t)h;
    return in->alive;
}

void vf_view_camera(vf_view *v, const float *vp, const float *eye, float near_m, float far_m) {
    memcpy(v->vp, vp, sizeof(v->vp));
    memcpy(v->eye, eye, sizeof(v->eye));
    v->eye[3] = near_m; // xyz — глаз, w — ближняя планка
    v->far_m = far_m;
}

// ---------------------------------------------------------------- геометрия

static int32_t add_shape(vf_view *v, const void *verts, VkDeviceSize vsize,
                         const uint32_t *idx, uint32_t ni, int lines) {
    if (v->n_shapes >= MAX_SHAPES) return -1;
    gpu_shape *g = &v->shapes[v->n_shapes];
    memset(g, 0, sizeof(*g));
    g->is_lines = lines;
    VkDeviceSize bytes = vsize;
    if (!make_buffer(v, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &g->vbuf, &g->vmem,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return -1;
    void *p = NULL;
    vkMapMemory(v->dev, g->vmem, 0, VK_WHOLE_SIZE, 0, &p);
    memcpy(p, verts, (size_t)vsize);
    vkUnmapMemory(v->dev, g->vmem);

    if (!lines) {
        VkDeviceSize isize = sizeof(uint32_t) * ni;
        if (!make_buffer(v, isize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &g->ibuf, &g->imem,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            return -1;
        vkMapMemory(v->dev, g->imem, 0, VK_WHOLE_SIZE, 0, &p);
        memcpy(p, idx, (size_t)isize);
        vkUnmapMemory(v->dev, g->imem);
        g->n_idx = ni;
    } else {
        g->n_idx = (uint32_t)(vsize / (sizeof(float) * 3));
    }
    return v->n_shapes++;
}

int32_t vf_view_add_mesh(vf_view *v, const float *pos, const float *nrm, uint32_t nv,
                         const uint32_t *idx, uint32_t ni) {
    if (nv == 0 || ni == 0 || (uint64_t)nv > 4u * 1024u * 1024u) {
        fprintf(stderr, "[view] меш странного размера: %u вершин, %u индексов\n", nv, ni);
        return -1;
    }
    float *verts = malloc(sizeof(float) * 6 * nv);
    if (!verts) {
        fprintf(stderr, "[view] нет памяти под меш на %u вершин\n", nv);
        return -1;
    }
    for (uint32_t i = 0; i < nv; i++) {
        memcpy(verts + i * 6, pos + i * 3, sizeof(float) * 3);
        memcpy(verts + i * 6 + 3, nrm + i * 3, sizeof(float) * 3);
    }
    int32_t id = add_shape(v, verts, sizeof(float) * 6 * nv, idx, ni, 0);
    free(verts);
    return id;
}

int32_t vf_view_add_lines(vf_view *v, const float *pts, uint32_t n_lines) {
    return add_shape(v, pts, sizeof(float) * 3 * n_lines * 2, NULL, 0, 1);
}

// ---------------------------------------------------------------- кадр

void vf_view_begin(vf_view *v, float r, float g, float b) {
    vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, UINT64_MAX);
    uint32_t img = 0;
    VkResult ar = vkAcquireNextImageKHR(v->dev, v->sc, UINT64_MAX, v->sem_ready, VK_NULL_HANDLE, &img);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(v->dev);
        drop_swapchain(v);
        make_swapchain(v);
        make_depth(v);
        make_framebuffers(v);
        ar = vkAcquireNextImageKHR(v->dev, v->sc, UINT64_MAX, v->sem_ready, VK_NULL_HANDLE, &img);
    }
    if (ar != VK_SUCCESS) return;
    v->cur_img = img;
    vkResetFences(v->dev, 1, &v->fence);
    vkResetCommandBuffer(v->cmd, 0);

    float *u = (float *)v->ubo_ptr;
    memcpy(u, v->vp, sizeof(v->vp));
    memcpy(u + 16, v->eye, sizeof(float) * 4);
    u[20] = 0.55f; u[21] = 0.60f; u[22] = 0.72f; u[23] = v->far_m; // свет + дальняя планка

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(v->cmd, &bi);
    /* Физика отправляет такт асинхронно и не ждёт его: без барьера
     * вершинный шейдер может прочитать буфер тел, не увидев записи compute. */
    VkMemoryBarrier body_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    body_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    body_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &body_barrier, 0,
                         NULL, 0, NULL);
    VkClearValue clear[2];
    memset(clear, 0, sizeof(clear));
    clear[0].color.float32[0] = r;
    clear[0].color.float32[1] = g;
    clear[0].color.float32[2] = b;
    clear[0].color.float32[3] = 1.0f;
    clear[1].depthStencil.depth = 1.0f;
    VkRenderPassBeginInfo rp = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = v->rp;
    rp.framebuffer = v->fb[img];
    rp.renderArea.extent = v->extent;
    rp.clearValueCount = 2;
    rp.pClearValues = clear;
    vkCmdBeginRenderPass(v->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {0, 0, (float)v->extent.width, (float)v->extent.height, 0.0f, 1.0f};
    VkRect2D scissor = {{0, 0}, v->extent};
    vkCmdSetViewport(v->cmd, 0, 1, &viewport);
    vkCmdSetScissor(v->cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->lay_mesh, 0, 1, &v->dset, 0, NULL);
    v->recording = 1;
}

void vf_view_set_bodies(vf_view *v, void *buffer, uint32_t count) {
    if (!v || !buffer) return;
    v->body_buf = (VkBuffer)buffer;
    v->body_n = count;
    VkDescriptorBufferInfo dbi = {v->body_buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wd = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet = v->dset;
    wd.dstBinding = 1;
    wd.descriptorCount = 1;
    wd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wd.pBufferInfo = &dbi;
    vf_ctx_lock(v->core);
    vkUpdateDescriptorSets(v->dev, 1, &wd, 0, NULL);
    vf_ctx_unlock(v->core);
}

void vf_view_draw(vf_view *v, int32_t mesh, uint32_t body, float alpha, const float *tint) {
    if (!v->recording || mesh < 0 || mesh >= v->n_shapes) return;
    gpu_shape *g = &v->shapes[mesh];
    if (g->is_lines) return;
    vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipe_mesh);
    /* поза — в буфере тел, здесь только «какое тело» и alpha сглаживания */
    float pick[4] = {(float)body, alpha, 0.0f, 0.0f};
    vkCmdPushConstants(v->cmd, v->lay_mesh, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, 16, tint);
    vkCmdPushConstants(v->cmd, v->lay_mesh, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       16, 16, pick);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(v->cmd, 0, 1, &g->vbuf, &off);
    vkCmdBindIndexBuffer(v->cmd, g->ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(v->cmd, g->n_idx, 1, 0, 0, 0);
}

void vf_view_draw_lines(vf_view *v, int32_t lines, const float *model, const float *tint) {
    if (!v->recording || lines < 0 || lines >= v->n_shapes) return;
    gpu_shape *g = &v->shapes[lines];
    if (!g->is_lines) return;
    vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipe_line);
    vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->lay_line, 0, 1, &v->dset, 0, NULL);
    vkCmdPushConstants(v->cmd, v->lay_line, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, 64, model);
    vkCmdPushConstants(v->cmd, v->lay_line, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       64, 16, tint);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(v->cmd, 0, 1, &g->vbuf, &off);
    vkCmdDraw(v->cmd, g->n_idx, 1, 0, 0);
}

// ---------------------------------------------------------------- оверлей стадий
//
// Текст рисуется встроенным растровым шрифтом 5x7: каждый включённый пиксель
// знака — квад. Примитивов много, зато нет ни SDL_ttf, ни файлов шрифтов, и
// оверлей работает везде, где работает окно.

static void ui_quad(vf_view *v, float x, float y, float w, float h, const float *c) {
    if (!v->ui_ptr || v->ui_n + 6 > v->ui_cap) return;
    float W = (float)v->extent.width, H = (float)v->extent.height;
    float x0 = 2.0f * x / W - 1.0f;
    float y0 = 2.0f * y / H - 1.0f;
    float x1 = 2.0f * (x + w) / W - 1.0f;
    float y1 = 2.0f * (y + h) / H - 1.0f;
    const float px[6][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y0}, {x1, y1}, {x0, y1}};
    for (int i = 0; i < 6; i++) {
        float *p = v->ui_ptr + (size_t)v->ui_n * 6;
        p[0] = px[i][0];
        p[1] = px[i][1];
        p[2] = c[0];
        p[3] = c[1];
        p[4] = c[2];
        p[5] = c[3];
        v->ui_n++;
    }
}

/* Строка ASCII-текста. c — цвет, scale — целое увеличение пикселя шрифта. */
static void ui_text(vf_view *v, float x, float y, float scale, const float *c, const char *t) {
    for (; *t; t++) {
        unsigned char ch = (unsigned char)*t;
        if (ch < VF_FONT_FIRST || ch > VF_FONT_LAST) ch = '?';
        const unsigned char *g = &vf_font5x7[(ch - VF_FONT_FIRST) * 5];
        for (int col = 0; col < 5; col++) {
            /* Вертикальные серии бит рисуем одним квадом, а не по пикселю:
             * вершин втрое меньше при том же виде. */
            int row = 0;
            while (row < 7) {
                if ((g[col] >> row) & 1) {
                    int run = 1;
                    while (row + run < 7 && ((g[col] >> (row + run)) & 1)) run++;
                    ui_quad(v, x + col * scale, y + row * scale, scale, run * scale, c);
                    row += run;
                } else {
                    row++;
                }
            }
        }
        x += 6.0f * scale;
    }
}

/* Три числа стадии одной строкой: «мин / сред / макс». При крупных значениях
 * переходим на одну цифру после точки — иначе триплет налезает на долю. */
static void fmt_triplet(char *buf, size_t n, float mn, float me, float mx) {
    if (mx < 10.0f) {
        snprintf(buf, n, "%.2f / %.2f / %.2f", (double)mn, (double)me, (double)mx);
    } else {
        snprintf(buf, n, "%.1f / %.1f / %.1f", (double)mn, (double)me, (double)mx);
    }
}

/* График истории: n отсчётов на w пикселей, верх — vmax. Столбики 1-2 px без
 * заливки: график — фон под цифрами, а не цветное пятно; новых конвейеров не
 * нужно, всё те же квады из ui_quad. g1/g2 — линии-ориентиры (60/30 fps,
 * номинал/половина tps), чтобы просадку можно было читать без линейки. */
static void ui_graph(vf_view *v, float x, float y, float w, float h,
                     const float *s, uint32_t n, float vmax,
                     const float *bar, const float *guide, float g1, float g2) {
    if (n == 0 || vmax <= 0.0f) return;
    float fr[4] = {guide[0], guide[1], guide[2], 0.22f};
    ui_quad(v, x, y, w, 1.0f, fr);
    ui_quad(v, x, y + h - 1.0f, w, 1.0f, fr);
    ui_quad(v, x, y, 1.0f, h, fr);
    ui_quad(v, x + w - 1.0f, y, 1.0f, h, fr);
    float gl[4] = {guide[0], guide[1], guide[2], 0.38f};
    if (g1 > 0.0f && g1 < vmax) ui_quad(v, x, y + h - h * g1 / vmax, w, 1.0f, gl);
    if (g2 > 0.0f && g2 < vmax) ui_quad(v, x, y + h - h * g2 / vmax, w, 1.0f, gl);
    float bw = w / (float)n;
    float bw_draw = bw > 1.6f ? bw - 0.6f : 1.0f;
    for (uint32_t i = 0; i < n; i++) {
        float val = s[i];
        if (!(val > 0.0f)) continue;
        float bh = h * val / vmax;
        if (bh > h) bh = h;
        if (bh < 1.0f) bh = 1.0f;
        ui_quad(v, x + bw * (float)i, y + h - bh, bw_draw, bh, bar);
    }
}

/* Частота одной строкой: целые от 10 и выше, десятые ниже. На просадках
 * «2 fps» и «0 fps» скрывали бы разницу между 0.3 и 9 — а именно там и
 * интересно. */
static void fmt_rate(char *buf, size_t n, float v) {
    if (v >= 10.0f) {
        snprintf(buf, n, "%.0f", (double)v);
    } else {
        snprintf(buf, n, "%.1f", (double)v);
    }
}

static void draw_perf_overlay(vf_view *v) {
    if (!v->pipe_ui || !v->ui_ptr || !v->perf_show || !v->recording) return;
    v->ui_n = 0;

    const float bg[4] = {0.035f, 0.05f, 0.075f, 0.88f};
    const float title[4] = {0.78f, 0.86f, 1.0f, 1.0f};
    const float head[4] = {0.55f, 0.62f, 0.72f, 1.0f};
    const float text[4] = {0.90f, 0.93f, 0.97f, 1.0f};
    static const float pal[8][3] = {
        {0.95f, 0.42f, 0.35f}, {0.98f, 0.68f, 0.30f}, {0.92f, 0.86f, 0.35f},
        {0.48f, 0.82f, 0.44f}, {0.36f, 0.78f, 0.80f}, {0.42f, 0.60f, 0.95f},
        {0.70f, 0.55f, 0.95f}, {0.92f, 0.50f, 0.75f},
    };

    const float sc = 2.0f, lh = 18.0f;
    /* Панель шире прежней: «мин / сред / макс» идут одной группой — три числа
     * рядом читаются как разброс стадии, а не как три несвязанных колонки. */
    const float x0 = 10.0f, y0 = 10.0f, w = 620.0f;
    /* Верх панели: заголовок, две строки частот, подписи и два графика
     * (250x40 каждый). Таблица стадий идёт под ними. */
    const float yrate1 = y0 + 30.0f;
    const float yrate2 = y0 + 48.0f;
    const float ycap = y0 + 72.0f;
    const float ygraph = y0 + 88.0f;
    const float gh = 40.0f, gw = 250.0f;
    const float yhead = ygraph + gh + 18.0f;
    uint32_t nrows = v->perf_n;
    /* Шапка + строки стадий + строка ИТОГО такта (+ строка тел, если есть).
     * Пока такта не было — две строки: строка частот уже нарисована, под ней
     * «waiting for a physics tick». */
    float h = (yhead - y0)
            + lh * (float)(nrows ? nrows + 2 + (v->perf_scene_ok ? 1 : 0) : 2)
            + 8.0f;
    ui_quad(v, x0, y0, w, h, bg);

    char line[96];
    if (v->perf_tick_max_ms > 0.0f) {
        char tri[48];
        if (v->perf_tick_max_ms < 10.0f) {
            snprintf(tri, sizeof tri, "%.2f/%.2f/%.2f",
                     (double)v->perf_tick_min_ms, (double)v->perf_tick_mean_ms,
                     (double)v->perf_tick_max_ms);
        } else {
            snprintf(tri, sizeof tri, "%.1f/%.1f/%.1f",
                     (double)v->perf_tick_min_ms, (double)v->perf_tick_mean_ms,
                     (double)v->perf_tick_max_ms);
        }
        snprintf(line, sizeof line, "GPU STAGES  tick %s ms  F3 toggle", tri);
    } else {
        snprintf(line, sizeof line, "GPU STAGES   no tick yet   F3 toggle");
    }
    ui_text(v, x0 + 10, y0 + 8, sc, title, line);

    /* Частоты — первой строкой: чаще всего хочется знать именно это. Среднее
     * плюс крайние за то же окно: одно среднее прячет просадку на глаз. */
    if (v->rates_ok && v->rates_frames > 0) {
        char fe[16], fn[16], fx[16], te[16], tn[16], tx[16];
        fmt_rate(fe, sizeof fe, v->fps_mean);
        fmt_rate(fn, sizeof fn, v->fps_min);
        fmt_rate(fx, sizeof fx, v->fps_max);
        fmt_rate(te, sizeof te, v->tps_mean);
        fmt_rate(tn, sizeof tn, v->tps_min);
        fmt_rate(tx, sizeof tx, v->tps_max);
        snprintf(line, sizeof line, "fps %s (min %s max %s)  frame %.1f ms",
                 fe, fn, fx, (double)v->frame_mean_ms);
        ui_text(v, x0 + 10, yrate1, sc, title, line);
        /* Вторая строка — tps и опорное 60 Гц: на 620 px панели всё одной
         * строкой не встаёт, а переносить числа по разным строкам нельзя —
         * fps/tps должны читаться рядом. */
        snprintf(line, sizeof line, "tps %s (min %s max %s)  60 Hz = 16.7 ms",
                 te, tn, tx);
        ui_text(v, x0 + 10, yrate2, sc, head, line);
    } else {
        ui_text(v, x0 + 10, yrate1, sc, head, "fps -- (min -- max --)  frame -- ms");
        ui_text(v, x0 + 10, yrate2, sc, head, "tps -- (min -- max --)");
    }

    /* Графики: ориентиры — 60/30 fps (обычный бюджет кадра) и номинал/половина
     * tps. Но если машина сильно ниже (софтверный рендер, тяжёлая сцена),
     * фиксированная шкала прижала бы все столбики к нулю и график перестал бы
     * что-либо показывать — тогда автомасштаб по максимуму истории. */
    {
        float fmax = 0.0f, tmax = 0.0f;
        for (uint32_t i = 0; i < v->hist_n; i++) {
            if (v->hist_fps[i] > fmax) fmax = v->hist_fps[i];
            if (v->hist_tps[i] > tmax) tmax = v->hist_tps[i];
        }
        float fs, fg1, fg2;
        if (fmax >= 30.0f) {
            fs = 60.0f;
            while (fs < fmax * 1.05f && fs < 1000.0f) fs += 30.0f;
            fg1 = 60.0f;
            fg2 = 30.0f;
        } else {
            fs = 5.0f;
            while (fs < fmax * 1.15f && fs < 1000.0f) fs += 5.0f;
            fg1 = fs;
            fg2 = fs * 0.5f;
        }
        float ts;
        if (tmax >= 10.0f) {
            ts = 20.0f;
            while (ts < tmax * 1.05f && ts < 500.0f) ts += 10.0f;
        } else {
            ts = 3.0f;
            while (ts < tmax * 1.15f && ts < 500.0f) ts += 3.0f;
        }
        float bar1[4] = {text[0], text[1], text[2], 0.55f};
        float bar2[4] = {text[0], text[1], text[2], 0.55f};
        float gx2 = x0 + 12.0f + gw + 16.0f;
        ui_graph(v, x0 + 12.0f, ygraph, gw, gh, v->hist_fps, v->hist_n, fs,
                 bar1, head, fg1, fg2);
        ui_graph(v, gx2, ygraph, gw, gh, v->hist_tps, v->hist_n, ts,
                 bar2, head, ts, ts * 0.5f);
        snprintf(line, sizeof line, "fps 0-%.0f", (double)fs);
        ui_text(v, x0 + 12.0f, ycap, sc, head, line);
        snprintf(line, sizeof line, "tps 0-%.0f", (double)ts);
        ui_text(v, gx2, ycap, sc, head, line);
    }

    if (nrows == 0) {
        /* Такт ещё не закрыт (мир спит или диагностику только включили) — это
         * не «нет данных», а именно «такт не приходил», и так и говорим. */
        ui_text(v, x0 + 12, yhead + lh, sc, head, "waiting for a physics tick");
        vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipe_ui);
        VkDeviceSize off0 = 0;
        vkCmdBindVertexBuffers(v->cmd, 0, 1, &v->ui_vbuf, &off0);
        vkCmdDraw(v->cmd, v->ui_n, 1, 0, 0);
        v->ui_n = 0;
        return;
    }
    ui_text(v, x0 + 12, yhead, sc, head, "stage");
    ui_text(v, x0 + 210, yhead, sc, head, "min / mean / max");
    ui_text(v, x0 + 470, yhead, sc, head, "share");

    for (uint32_t i = 0; i < v->perf_n; i++) {
        const vf_perf_row *r = &v->perf_rows[i];
        float y = yhead + lh * (float)(i + 1);
        float share = r->share;
        if (share < 0.0f) share = 0.0f;
        if (share > 100.0f) share = 100.0f;
        /* Цвет — по имени стадии (сумма букв): у solve_contacts и solve_apply
         * общий префикс, но полосы должны различаться. */
        unsigned hsh = 0;
        for (const char *p = r->name; *p; p++) hsh = hsh * 31u + (unsigned char)*p;
        float bar[4] = {pal[hsh % 8][0], pal[hsh % 8][1], pal[hsh % 8][2], 0.30f};
        float bw = (w - 24.0f) * share / 100.0f;
        if (bw > 1.0f) ui_quad(v, x0 + 12, y, bw, lh - 5.0f, bar);

        char name[24];
        snprintf(name, sizeof name, "%.18s", r->name);
        ui_text(v, x0 + 12, y + 4, sc, text, name);
        fmt_triplet(line, sizeof line, r->min_ms, r->mean_ms, r->max_ms);
        ui_text(v, x0 + 210, y + 4, sc, text, line);
        snprintf(line, sizeof line, "%5.1f%%", (double)share);
        ui_text(v, x0 + 470, y + 4, sc, text, line);
    }

    /* ИТОГО такта: та же тройка чисел, что в консоли. По ней видно, сколько
     * такт стоит целиком и насколько он неровный — стадии её не заменяют. */
    {
        float y = yhead + lh * (float)(nrows + 1);
        ui_text(v, x0 + 12, y + 4, sc, title, "total tick");
        fmt_triplet(line, sizeof line, v->perf_tick_min_ms, v->perf_tick_mean_ms,
                    v->perf_tick_max_ms);
        ui_text(v, x0 + 210, y + 4, sc, title, line);
        ui_text(v, x0 + 470, y + 4, sc, head, "100.0%");
    }

    /* Объём работы: просадка стадии без числа тел ничего не объясняет —
     * 30 мс на 13 телах и на 60 телах это разные вещи. Спящие считаются как
     * всего минус бодрствующие (отдельного счётчика для них нет). */
    if (v->perf_scene_ok) {
        float y = yhead + lh * (float)(nrows + 2);
        uint32_t tot = v->perf_scene.total;
        uint32_t aw = v->perf_scene.awake > tot ? tot : v->perf_scene.awake;
        snprintf(line, sizeof line, "bodies %u  awake %u  asleep %u  contacts %u",
                 (unsigned)tot, (unsigned)aw, (unsigned)(tot - aw),
                 (unsigned)v->perf_scene.contacts);
        ui_text(v, x0 + 12, y + 4, sc, head, line);
    }

    vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipe_ui);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(v->cmd, 0, 1, &v->ui_vbuf, &off);
    vkCmdDraw(v->cmd, v->ui_n, 1, 0, 0);
    v->ui_n = 0;
}

void vf_view_perf(vf_view *v, const vf_perf_row *rows, uint32_t n,
                  float tick_min_ms, float tick_mean_ms, float tick_max_ms,
                  const vf_scene_counts *scene, const vf_rates *rates) {
    if (!v) return;
    /* rows == NULL — оверлей выключен; непустой указатель при n == 0 — включён,
     * но такта ещё не было (панель так и скажет). */
    v->perf_show = rows != NULL;
    v->perf_n = 0;
    v->perf_tick_min_ms = tick_min_ms;
    v->perf_tick_mean_ms = tick_mean_ms;
    v->perf_tick_max_ms = tick_max_ms;
    v->perf_scene_ok = scene != NULL;
    if (scene) v->perf_scene = *scene;
    /* Частоты и история копируются внутрь окна: указатели истории живут только
     * на время вызова, а рисуем мы их в vf_view_end. */
    v->rates_ok = rates != NULL;
    v->hist_n = 0;
    if (rates) {
        v->fps_mean = rates->fps_mean;
        v->fps_min = rates->fps_min;
        v->fps_max = rates->fps_max;
        v->tps_mean = rates->tps_mean;
        v->tps_min = rates->tps_min;
        v->tps_max = rates->tps_max;
        v->frame_mean_ms = rates->frame_mean_ms;
        v->rates_frames = rates->frames;
        uint32_t nf = rates->fps_hist_n;
        if (nf > VF_HIST_MAX) nf = VF_HIST_MAX;
        if (rates->fps_hist && nf) {
            memcpy(v->hist_fps, rates->fps_hist, (size_t)nf * sizeof(float));
        }
        uint32_t nt = rates->tps_hist_n;
        if (nt > VF_HIST_MAX) nt = VF_HIST_MAX;
        if (rates->tps_hist && nt) {
            memcpy(v->hist_tps, rates->tps_hist, (size_t)nt * sizeof(float));
        }
        v->hist_n = nf < nt ? nf : nt;
    }
    if (!rows || n == 0) return;
    if (n > VF_PERF_MAX_ROWS) n = VF_PERF_MAX_ROWS;
    memcpy(v->perf_rows, rows, (size_t)n * sizeof(*rows));
    v->perf_n = n;
}

void vf_view_end(vf_view *v) {
    if (!v->recording) return;
    draw_perf_overlay(v);
    vkCmdEndRenderPass(v->cmd);
    vkEndCommandBuffer(v->cmd);
    VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &v->sem_ready;
    si.pWaitDstStageMask = &wait;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &v->sem_done;
    /* Очередь общая с физикой: submit и present — под замком ядра.
     * Fence ждём вне замка (см. vf_view_begin). */
    vf_ctx_lock(v->core);
    VkResult sres = vkQueueSubmit(v->queue, 1, &si, v->fence);
    vf_ctx_unlock(v->core);
    if (sres != VK_SUCCESS) return;
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &v->sem_done;
    pi.swapchainCount = 1;
    pi.pSwapchains = &v->sc;
    // pImageIndices обязателен: без него драйвер разыменовывает NULL и падает
    // в wsi_common_queue_present.
    pi.pImageIndices = &v->cur_img;
    vf_ctx_lock(v->core);
    VkResult pr = vkQueuePresentKHR(v->queue, &pi);
    vf_ctx_unlock(v->core);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(v->dev);
        drop_swapchain(v);
        make_swapchain(v);
        make_depth(v);
        make_framebuffers(v);
    }
    v->recording = 0;
}
