## 初始化解混矩阵
随机初始化解混矩阵。

假设我们有3个EEG电极，那么解混矩阵 $\mathbf{W}$ 的维度是 $3 \times 3$：
使用标准正态分布生成随机数初始化解混矩阵：
  
$$\mathbf{W} = \begin{pmatrix}
0.5 & -0.3 & 0.8 \\
-0.2 & 0.6 & -0.1 \\
0.3 & -0.4 & 0.5
\end{pmatrix}
$$

## FastICA算法

### 非线性函数   $g $ 和其导数 $g' $ 的作用
在FastICA算法中，非线性函数 $ g $ 和其导数 $ g' $ 用于最大化独立成分的非高斯性，这是分离信号的核心步骤。

#### 非线性函数 $ g $ 的作用
非线性函数 $ g $ 用于增强信号的非高斯性，使得信号更容易被分离出来。常用的非线性函数包括：

- 对数函数（Logarithm）：  
  $$g(y) = \log(\cosh(y))
  $$
  导数：
  $$g'(y) = \tanh(y)
  $$

- 双曲正切函数（Hyperbolic Tangent）：
  $$g(y) = \tanh(y)
  $$
  导数：
  $$g'(y) = 1 - \tanh^2(y)
  $$

- 三次函数（Cubic Function）：
  $$g(y) = y^3
  $$
  导数：
  $$g'(y) = 3y^2
  $$

#### 导数 $ g' $ 的作用
非线性函数的导数 $ g' $ 在更新解混矩阵的过程中起到关键作用。具体来说，导数 $ g' $ 用于计算信号的期望值，这些期望值在优化步骤中用于调整解混矩阵。

### 具体步骤
1. **初始化解混矩阵 $\mathbf{W}$**：
   - 随机初始化解混矩阵。

2. **迭代更新解混矩阵**：
   - 对于每一个独立成分，进行如下更新：
     $$\mathbf{w}_{\text{new}} = \mathbf{X}_w \cdot g(\mathbf{W}^T \mathbf{X}_w) - \text{mean}[g'(\mathbf{W}^T \mathbf{X}_w)] \cdot \mathbf{W}
     $$
   - 其中：
    $$g(\mathbf{W}^T \mathbf{X}_w) $$ 是对信号应用非线性函数 $ g $ 后的结果，用于增强非高斯性。
    $$g'(\mathbf{W}^T \mathbf{X}_w) $$ 是对信号应用非线性函数导数 $ g' $ 后的结果，用于计算信号的期望值。
  在这段公式中，mean 表示对括号内的向量进行均值操作
3. **标准化**：
   - 对更新后的向量进行标准化：
  
![image](https://github.com/user-attachments/assets/0df1b495-9455-42bf-81de-09b1cb5aa63c)


4. **正交化**：
   - 确保不同独立成分之间是正交的。

### 示例
假设我们使用双曲正切函数 $ g(y) = \tanh(y) $ 进行计算：

1. **非线性变换**：
   $$g(\mathbf{W}^T \mathbf{X}_w) = \tanh(\mathbf{W}^T \mathbf{X}_w)
   $$

2. **导数变换**：
   $$g'(\mathbf{W}^T \mathbf{X}_w) = 1 - \tanh^2(\mathbf{W}^T \mathbf{X}_w)
   $$

3. **迭代更新**：
   $$\mathbf{w}_{\text{new}} = \mathbf{X}_w \cdot \tanh(\mathbf{W}^T \mathbf{X}_w) - \text{mean}[1 - \tanh^2(\mathbf{W}^T \mathbf{X}_w)] \cdot \mathbf{W}
   $$
   
![image](https://github.com/user-attachments/assets/01cb9728-ed56-4d86-81d5-dfc7939a85ad)
![image](https://github.com/user-attachments/assets/7d53d93d-0b74-468a-aea5-524b86592221)
