#include <atb/types.h>
#include <acl/acl.h>
#include "memorypool.h"
#include "utils/log.h"
#include "utils/utils.h"

constexpr size_t POOL_SIZE = 104857600; // Alloceted memory 100 MiB.

MemoryPool::MemoryPool(size_t poolSize = POOL_SIZE)
{
    CHECK_RET(aclrtMalloc(&baseMemPtr_, poolSize, ACL_MEM_MALLOC_HUGE_FIRST),
              "malloc huge size memrory " + std::to_string(poolSize) + " bytes fail");
    curMemPtr_ = baseMemPtr_;
    remainSize_ = poolSize;
}

MemoryPool::~MemoryPool()
{
    if (baseMemPtr_ != nullptr) {
        CHECK_RET(aclrtFree(baseMemPtr_), "free huge memory fail");
    }
    LOG_INFO("release MemoryPool success");
}

uint64_t MemoryPool::GenerateBlocksId()
{
    return static_cast<uint64_t>(id_.fetch_add(1, std::memory_order_relaxed));
}

void MemoryPool::AllocateBlock(uint32_t size, int &blockId)
{
    // 获取互斥锁，确保线程安全
    std::unique_lock<std::mutex> lock(blockMutex_);

    // 计算对齐后的大小：32字节对齐 + 额外32字节开销
    // 31 = 32-1，用于32字节对齐；额外32字节可能用于元数据或填充
    size_t alignSize = ((size + 31) & ~31) + 32;
    
    // 策略1：尝试从空闲块中复用
    // 遍历所有空闲内存块，寻找大小足够的内存块
    for (auto it = freeBlocks_.begin() ; it != freeBlocks_.end() ; it++) {
        if (it->second.blockSize >= alignSize) {
            // 找到合适大小的空闲块
            blockId = it->second.blockId;
            // 将块从空闲表移动到已使用表
            usedBlocks_.insert(*it);
            freeBlocks_.erase(it);
            LOG_INFO("find free block id " + std::to_string(blockId) + " to allocate");
            return;
        }
    }
    
    // 策略2：从剩余内存中分配新块
    // 检查剩余内存是否足够
    if (remainSize_ > alignSize) {
        // 生成新的块ID
        blockId = GenerateBlocksId();
        
        // 计算64字节对齐的当前内存地址
        // 63 = 64-1，用于64字节对齐
        uint64_t curMemPtrAlign = (reinterpret_cast<uint64_t>(curMemPtr_) + 63) & ~ 63;
        
        // 更新剩余大小，减去对齐产生的填充
        remainSize_ -= (curMemPtrAlign - reinterpret_cast<uint64_t>(curMemPtr_));
        // 更新当前内存指针到对齐后的地址
        curMemPtr_ = reinterpret_cast<void *>(curMemPtrAlign);

        // 创建新的内存块并添加到已使用表
        MemoryBlock block = {blockId, alignSize, curMemPtr_};
        usedBlocks_.insert({blockId, block});
        
        // 更新剩余大小和当前指针
        remainSize_ -= alignSize;
        curMemPtr_ = reinterpret_cast<uint8_t *>(curMemPtr_) + alignSize;
        
        LOG_INFO("allocate block id " + std::to_string(blockId) + " for size " + std::to_string(alignSize));
        return;
    }
    
    // 内存不足，分配失败
    LOG_ERROR("allocate block fail");
}

void MemoryPool::FreeBlock(int blockId)
{
    std::unique_lock<std::mutex> lock(blockMutex_);

    if (blockId < 0) {
        LOG_INFO("skip over the invalid block id " + std::to_string(blockId));
        return ;
    }
    auto it = usedBlocks_.find(blockId);
    if (it != usedBlocks_.end()) {
        freeBlocks_.insert(*it);
        usedBlocks_.erase(it);
    } else {
        LOG_ERROR("Double free block id " + std::to_string(blockId));
    }
}

void MemoryPool::GetBlockPtr(int blockId, void *&addr)
{
    std::unique_lock<std::mutex> lock(blockMutex_);

    if (blockId < 0) {
        LOG_INFO("Invalid block id " + std::to_string(blockId) + "to get ptr");
        return ;
    }
    auto it = usedBlocks_.find(blockId);
    if (it != usedBlocks_.end()) {
        addr = it->second.address;
    } else {
        LOG_ERROR("Get block address error, block id " + std::to_string(blockId));
    }
}