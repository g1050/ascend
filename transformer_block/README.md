### 介绍
本代码框架主要是示范如何使用ATB算子和aclnn算子构建一个小模型的执行框架。涉及到模型的构建，模型层的构建，以及层中如何使用ATB的原生算子，plugin算子和图算子或者是aclnn算子。

### 目录介绍
aclnn: aclnn统一接入ATB流程的代码<br>
atb: atb的图算子创建<br>
model: 定义一个模型<br>
utils: 用到的辅助函数<br>

### 算子类型
ATB：原生算子，plugin算子和图算子
aclnn：Gelu算子

### 使用教程
 - 编译<br>
   ```sh
    > bash build.sh
    ```

 - 执行<br>
    ```sh
    > cd build
    > ./test_model
    ```