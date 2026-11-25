/* libvoxelforge_core — Vulkan core, plain C, C ABI. См. voxelforge_core.h.
 *
 * Только Vulkan: нет логики мира, нет физики, нет шейдерных констант.
 * Модель исполнения простая и достаточная для headless-демок: записали пачку
 * dispatch'ей в один command buffer, отправили, дождались fence, прочитали
 * результат из host-visible памяти. Никаких фреймов, swapchain и синхронизации
 * с окном — это появится вместе с окном, а не раньше.
 */

#include "voxelforge_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VF_MAX_BINDINGS 16
#define VF_MAX_SETS 8192

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

struct vf_ctx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t family;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorPool desc_pool;
    VkPhysicalDeviceMemoryProperties memp;
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

vf_status vf_init(vf_ctx **out) {
    vf_ctx *c = calloc(1, sizeof(*c));
    if (!c) return ERR(-1, "vmf: нет памяти под контекст");

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "voxelforge";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &c->instance) != VK_SUCCESS) {
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

    /* Первое устройство с очередью compute. */
    for (uint32_t i = 0; i < n && !c->phys; i++) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
        VkQueueFamilyProperties *qf = calloc(qn, sizeof(*qf));
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf);
        for (uint32_t q = 0; q < qn; q++) {
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                c->phys = devs[i];
                c->family = q;
                break;
            }
        }
        free(qf);
    }
    free(devs);
    if (!c->phys) {
        vkDestroyInstance(c->instance, NULL);
        free(c);
        return ERR(-4, "vulkan: нет устройства с compute-очередью");
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(c->phys, &props);
    snprintf(c->name, sizeof(c->name), "%s", props.deviceName);
    c->api = props.apiVersion;
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->memp);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c->family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    /* apiVersion 1.3, но без фич: ядру нужен только compute + storage buffers. */
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_FALSE; /* барьеры ставим старым API — работает везде */

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(c->phys, &dci, NULL, &c->dev) != VK_SUCCESS) {
        vkDestroyInstance(c->instance, NULL);
        free(c);
        return ERR(-5, "vulkan: не удалось создать логическое устройство");
    }
    vkGetDeviceQueue(c->dev, c->family, 0, &c->queue);

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->family;
    vkCreateCommandPool(c->dev, &pci, NULL, &c->pool);

    VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = c->pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    vkAllocateCommandBuffers(c->dev, &cbi, &c->cb);

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(c->dev, &fci, NULL, &c->fence);

    VkDescriptorPoolSize sz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VF_MAX_SETS * VF_MAX_BINDINGS};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = VF_MAX_SETS;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &sz;
    if (vkCreateDescriptorPool(c->dev, &dpci, NULL, &c->desc_pool) != VK_SUCCESS) {
        vf_shutdown(c);
        return ERR(-6, "vulkan: пул дескрипторов не создался");
    }

    *out = c;
    return OK;
}

void vf_shutdown(vf_ctx *c) {
    if (!c) return;
    if (c->dev) vkDeviceWaitIdle(c->dev);
    if (c->desc_pool) vkDestroyDescriptorPool(c->dev, c->desc_pool, NULL);
    if (c->fence) vkDestroyFence(c->dev, c->fence, NULL);
    if (c->pool) vkDestroyCommandPool(c->dev, c->pool, NULL);
    if (c->dev) vkDestroyDevice(c->dev, NULL);
    if (c->instance) vkDestroyInstance(c->instance, NULL);
    free(c);
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
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
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

vf_status vf_run(vf_ctx *c, vf_dispatch *list, uint32_t count) {
    if (!c) return ERR(-1, "нет контекста");
    if (count == 0) return OK;

    vkResetDescriptorPool(c->dev, c->desc_pool, 0);
    vkResetCommandBuffer(c->cb, 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c->cb, &bi) != VK_SUCCESS)
        return ERR(-2, "vulkan: command buffer не начался");

    for (uint32_t d = 0; d < count; d++) {
        vf_dispatch *dp = &list[d];
        if (!dp->pipeline) return ERR(-3, "dispatch без пайплайна");
        if (dp->n_buffers != dp->pipeline->n_buffers)
            return ERR(-4, "число буферов не совпадает с пайплайном");

        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = c->desc_pool;
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

        vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, dp->pipeline->pipe);
        vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, dp->pipeline->layout,
                                0, 1, &set, 0, NULL);
        if (dp->push_bytes)
            vkCmdPushConstants(c->cb, dp->pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, dp->push_bytes, dp->push);
        vkCmdDispatch(c->cb, dp->groups[0], dp->groups[1], dp->groups[2]);

        /* Шейдерная запись → шейдерное чтение следующего прохода. */
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }

    if (vkEndCommandBuffer(c->cb) != VK_SUCCESS)
        return ERR(-6, "vulkan: command buffer не закрылся");

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c->cb;
    vkResetFences(c->dev, 1, &c->fence);
    if (vkQueueSubmit(c->queue, 1, &si, c->fence) != VK_SUCCESS)
        return ERR(-7, "vulkan: submit провалился");
    if (vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return ERR(-8, "vulkan: fence не дождались");
    return OK;
}
