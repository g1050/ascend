#include "atb/atb_graph_op.h"
#include "utils/utils.h"

atb::Status CreateGraphOperationLN(atb::Operation **operation)
{
    // 构图流程
    // 图算子的输入a,b,c,d
    // 计算公式：(a+b) + (c+d)
    // 输入是4个参数，输出是1个参数，有3个add算子，中间产生的临时输出是2个
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 5; // 输入的tensor数量
    opGraph.outTensorNum = 1; // 输出的tensor数量
    opGraph.internalTensorNum = 1; // 中间产生的临时输出tensor数量
    opGraph.nodes.resize(2); // 图算子包含的节点数量

    // 普通枚举，可以直接使用
    enum InTensorId
    { // 定义各TensorID
        IN_TENSOR_X = 0,
        IN_TENSOR_GAMMA,
        IN_TENSOR_BETA,
        IN_TENSOR_MATMUL_WEIGHT,
        IN_TENSOR_MATMUL_BIAS,
        OUT_TENSOR_LN,
        OUT_TENSOR_LN_MATMUL,
    };

    size_t nodeId = 0;
    atb::Node &layerNode = opGraph.nodes.at(nodeId++);
    atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
    // atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam layerNormParam;
    const int32_t BEGIN_NORM_AXIS = 2;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.beginNormAxis = BEGIN_NORM_AXIS;
    auto status = atb::CreateOperation(layerNormParam,&layerNode.operation);
    CHECK_RET(status, "addParam CreateOperation failed. status: " + std::to_string(status));
    layerNode.inTensorIds = {IN_TENSOR_X,IN_TENSOR_GAMMA,IN_TENSOR_BETA};
    layerNode.outTensorIds = {OUT_TENSOR_LN};

    // 创建node:(a+b)
    // atb用来创建operation,operation可以组成graph
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.hasBias = true;
    param.outDataType = aclDataType::ACL_DT_UNDEFINED;
    param.enAccum = false;
    param.matmulType = atb::infer::LinearParam::MatmulType::MATMUL_UNDEFINED;
    param.quantMode = atb::infer::LinearParam::QuantMode::QUANT_UNDEFINED;
    // 创建operation需要两个参数，参数和operation二级指针
    status = atb::CreateOperation(param, &matmulNode.operation); // 每个node需要配置单算子对象实例
    CHECK_RET(status, "matmulParam CreateOperation failed. status: " + std::to_string(status));
    matmulNode.inTensorIds = {OUT_TENSOR_LN, IN_TENSOR_MATMUL_WEIGHT, IN_TENSOR_MATMUL_BIAS}; // 每个node需要配置输入的tensor id
    matmulNode.outTensorIds = {OUT_TENSOR_LN_MATMUL}; // 每个node需要配置输出的tensor id

    // 创建node:(c+d)
    // atb::infer::ElewiseParam addParam2;
    // addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    // status = atb::CreateOperation(addParam2, &addNode2.operation);
    // CHECK_RET(status, "addParam2 CreateOperation failed. status: " + std::to_string(status));
    // addNode2.inTensorIds = {OUT_TENSOR, IN_TENSOR_ADD};
    // addNode2.outTensorIds = {OUT_TENSOR_ADD};

    // // 创建node:(a+b)+(c+d)
    // atb::infer::ElewiseParam addParam3;
    // addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    // status = CreateOperation(addParam3, &addNode3.operation);
    // CHECK_RET(status, "addParam3 CreateOperation failed. status: " + std::to_string(status));
    // addNode3.inTensorIds = {ADD1_OUT, ADD2_OUT};
    // addNode3.outTensorIds = {ADD3_OUT};

    // 将graph添加到混合模型中
    status = atb::CreateOperation(opGraph, operation);
    CHECK_RET(status, "GraphParam CreateOperation failed. status: " + std::to_string(status));
    LOG_ERROR("完成创建(layerNorm)的图，图包含的节点数量：" + std::to_string(opGraph.nodes.size()));
    return atb::NO_ERROR;
}