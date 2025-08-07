#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"
#include "utils/log.h"

enum class TensorType
{
    INTERNAL_TENSOR = 0,
    NOT_INTERNAL_TENSOR,
};

// 图节点，每个Node表示一个Operation或者GraphOperation
struct Node
{
    // Node对应的operation或者graphOperation。
    atb::Operation *operation_ = nullptr;

    // Node的输入tensors
    atb::SVector<atb::Tensor *> inTensors_{};

    // Node的输出tensors
    atb::SVector<atb::Tensor *> outTensors_{};

    // Node的输出是中间tensor类型
    atb::SVector<TensorType> outTensorTypes_{};

    atb::VariantPack variantPack_{};

    uint64_t workspaceSize_ = 0;
    int workspaceBlockId_ = -1;
    void *workspace_ = nullptr;
};

// 所有的Node组成一个完整的图。
/**
 * 模型类
 * 管理完整的神经网络图，包括输入输出张量、节点执行、资源管理等
 * 负责模型的初始化、执行和资源释放
 */
class Model
{
public:
    /**
     * 输入张量ID枚举
     * 定义模型中各个输入张量的标识符
     */
    enum InTensorId : int
    { 
        IN_TENSOR_A = 0,    // 输入张量A
        IN_TENSOR_B,        // 输入张量B
        IN_TENSOR_C,        // 输入张量C
        IN_TENSOR_D,        // 输入张量D
        Mode_INPUT_SIZE,     // 输入张量总数
    };

    /**
     * 输出张量ID枚举
     * 定义模型中各个输出张量的标识符
     */
    enum OutTensorId : int
    {
        GLUE_OUT = 0,       // 输出张量（粘合输出）
        Mode_OUTPUT_SIZE,    // 输出张量总数
    };

    /**
     * 构造函数
     * @param modelName 模型名称，用于标识和日志记录
     */
    explicit Model(std::string &&modelName = "") : modelName_(std::move(modelName))
    {
        LOG_INFO("Create model: " + modelName_);
    }

    /**
     * 初始化模型资源
     * 设置设备ID，分配计算资源
     * @param deviceId 设备ID，指定运行设备
     */
    void InitResource(uint32_t deviceId);

    /**
     * 创建模型图
     * 构建完整的神经网络计算图
     */
    void CreateModelGraph();

    /**
     * 创建模型的输入张量
     * 初始化输入张量的描述和内存
     */
    void CreateModelInput();

    /**
     * 创建模型的输出张量
     * 初始化输出张量的描述和内存
     */
    void CreateModelOutput();

    /**
     * 执行模型推理
     * 运行完整的神经网络前向传播
     */
    void Execute();

    /**
     * 等待流执行完成
     * 同步计算流，确保所有操作完成
     */
    void WaitFinish();

    /**
     * 释放模型资源
     * 清理内存、流、上下文等资源
     */
    void FreeResource();

    // 模型的输入张量集合
    atb::SVector<atb::Tensor> model_inTensors_;

    // 模型的输出张量集合
    atb::SVector<atb::Tensor> model_outTensors_;

private:
    /**
     * 创建图操作层
     * @param nodeId 节点ID
     */
    void CreateGraphOpLayer(size_t nodeId);
    
    /**
     * 创建ACLNN操作层
     * @param nodeId 节点ID
     */
    void CreateAclnnOpLayer(size_t nodeId);

    /**
     * 构建节点变体包
     * 为节点准备输入输出张量
     * @param nodeId 节点ID
     */
    void BuildNodeVariantPack(int nodeId);
    
    /**
     * 执行单个节点
     * @param nodeId 节点ID
     * @return 执行状态
     */
    atb::Status ExecuteNode(int nodeId);
    
    /**
     * 创建工作空间缓冲区
     * 为节点分配临时计算空间
     * @param nodeId 节点ID
     * @param workspaceSizeNeeded 需要的工作空间大小
     */
    void CreateWorkspaceBuffer(int nodeId, int workspaceSizeNeeded);

    /**
     * 推理形状
     * 根据输入张量描述推断输出张量的形状
     * @param inTensorDescs 输入张量描述
     * @param outTensorDescs 输出张量描述
     * @return 推理状态
     */
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs);

    std::string modelName_;                    // 模型名称
    uint32_t deviceId_ = 1;                   // 设备ID，默认为1
    atb::Context *mode_context_ = nullptr;    // 模型上下文，管理计算资源
    aclrtStream model_stream_ = nullptr;      // 计算流，用于异步执行
    std::vector<Node> nodes_;                 // 节点集合，组成完整的计算图

    // 模型的中间张量，用于连接不同层之间的数据流
    // 注意：中间张量的顺序很重要，需要保持正确的数据流
    std::vector<atb::Tensor> internalTensors_;
};

#endif