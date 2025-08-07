#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "memory_env.h"

/**
 * 内存池类
 * 用于高效管理内存分配和释放，减少内存碎片化
 * 支持动态分配和回收内存块
 */
class MemoryPool {
public:
    /**
     * 构造函数
     * @param poolSize 内存池的总大小（字节）
     */
    explicit MemoryPool(size_t poolSize);
    
    /**
     * 析构函数
     * 释放所有分配的内存
     */
    ~MemoryPool();
    
    /**
     * 分配内存块
     * @param size 请求的内存块大小（字节）
     * @param blockId 输出参数，返回分配的内存块ID
     */
    void AllocateBlock(uint32_t size, int &blockId);
    
    /**
     * 释放内存块
     * @param blockId 要释放的内存块ID
     */
    void FreeBlock(int blockId);
    
    /**
     * 获取内存块指针
     * @param blockId 内存块ID
     * @param addr 输出参数，返回内存块的地址
     */
    void GetBlockPtr(int blockId, void *&addr);

private:
    /**
     * 生成唯一的块ID
     * @return 新的块ID
     */
    uint64_t GenerateBlocksId();
    
    std::atomic<uint64_t> id_ = 0;                    // 原子计数器，用于生成唯一ID
    std::mutex blockMutex_;                           // 互斥锁，保护内存块操作
    void *baseMemPtr_ = nullptr;                      // 内存池基地址指针
    void *curMemPtr_ = nullptr;                       // 当前可用内存指针
    int64_t remainSize_ = 0;                          // 剩余可用内存大小
    std::unordered_map<int, MemoryBlock> freeBlocks_; // 空闲内存块映射表
    std::unordered_map<int, MemoryBlock> usedBlocks_; // 已使用内存块映射表
};

#endif