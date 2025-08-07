#include "model/model2.h"
#include "memory/memory_utils.h"
#include <thread>
#include "utils/utils.h"

void ModelExecute(uint32_t deviceId, Model2 &model)
{
    // 初始化模型，创建需要的context，stream
    model.InitResource(deviceId);

    // 创建模型图
    model.CreateModelGraph();

    // 创建模型输入，并填入值
    model.CreateModelInput();

    // // 创建模型的输出大小
    model.CreateModelOutput();

    // 模型执行
    model.Execute();

    // 打印输出Tensor的值
    PrintOutTensorValue(model.model_outTensors_.at(0));
    LOG_ERROR("完成模型执行");
    // 资源释放
    model.FreeResource();
    LOG_ERROR("完成资源释放");
}

// constexpr size_t THREAD_SIZE = 5;
constexpr size_t THREAD_SIZE = 1;

int main()
{
    // AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret, "aclInit failed. ret: " + std::to_string(ret));

    // 创建内存池
    size_t poolSize = 104857600; // Alloceted memory 100 MiB.
    GetMemoryManager().CreateMemoryPool(poolSize);

    // 创建模型图
    std::vector<Model2> modelArray(THREAD_SIZE);

    std::vector<std::thread> threadArray(THREAD_SIZE);
    for (size_t i = 0; i < THREAD_SIZE; i++) {
        Model2 &model = modelArray.at(i);
        threadArray.at(i) = std::thread([i, &model]{ModelExecute(i, model);});
        // lambda表达式 执行ModelExecute函数，传入[i,&model]
    }
    for (size_t i = 0; i < THREAD_SIZE; i++) {
        threadArray.at(i).join();
    }

    aclFinalize();
    LOG_ERROR("完成aclFinalize");
    return 0;
}