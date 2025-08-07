#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <memory>
#include <vector>
#include "memorypool.h"

// 内存管理器类，负责管理多设备的内存池
class MemoryManager {
public:
    MemoryManager(); // 构造函数
    // 创建每个设备的内存池，poolSize为每个内存池的大小
    void CreateMemoryPool(size_t poolSize);
    // 获取当前设备的device id
    int32_t GetDeviceId();
    // 获取当前设备对应的内存池
    std::shared_ptr<MemoryPool> &GetMemoryPool();
    // 分配指定大小的内存块，返回blockId
    void AllocateBlock(uint32_t size, int &blockId);
    // 释放指定blockId的内存块
    void FreeBlock(int blockId);
    // 获取指定blockId的内存块指针
    void GetBlockPtr(int blockId, void *&addr);

private:
    // 存储每个设备的内存池
    std::vector<std::shared_ptr<MemoryPool>> memoryPools_;
};

MemoryManager &GetMemoryManager();

#endif