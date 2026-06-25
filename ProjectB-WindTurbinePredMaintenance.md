# **Project B**

**Wind Turbine Predictive Maintenance Platform - Student Guide**

## **0. 项目前言（必读）**

### **0.1 项目定位（核心区别于普通入门项目）**

本项目面向**大二、零机器学习基础**学生，选择风力发电机预测性维护作为工业场景，使用 Kaggle 经典 **Wind Turbine SCADA Dataset** 完成一套从数据分析、多特征回归建模、异常检测到 API 部署的端到端机器学习工程项目。

核心逻辑：本项目不追求复杂模型和竞赛排名，而是用一个非常直观的工业问题训练完整工程能力：**业务问题 → 数据理解 → 可视化分析 → 多特征工程 → 多元线性回归建模 → 残差异常检测 → API开发 → Docker容器化 → 云端部署 → 系统设计表达**。

### **0.2 项目最终简历亮点**

完成本项目后，简历可直接写入以下成果：

- Developed an Industrial IoT predictive maintenance system for wind turbines using SCADA sensor data

- Built a multiple linear regression asset-performance model to forecast expected power output using wind speed, theoretical power curve, and wind direction features

- Implemented residual-based anomaly detection to identify turbine underperformance and potential mechanical degradation

- Engineered interpretable maintenance features such as wind direction sine/cosine, wind speed squared, power efficiency, and residual ratio

- Developed a maintenance alert API with FastAPI

- Containerized the service with Docker and deployed it on AWS EC2

### **0.3 适配人群与求职价值**

适配：大二在读、零机器学习基础、准备 ML Engineer / SDE / Data Engineer 暑期实习学生。

优势：相比泰坦尼克号、鸢尾花、房价预测等传统入门 Demo，本项目具备真实工业场景：风力发电机、SCADA 传感器、设备亚健康、预测性维护、清洁能源。它既足够简单，适合第一阶段学习多元线性回归，又能扩展到 FastAPI、Docker、AWS，形成一套完整的工程化项目闭环。

### **0.4 整体技术路线**

Dataset → EDA业务数据分析 → 多特征工程 → Multiple Linear Regression基线模型 → 残差分析异常检测 → 模型评估 → FastAPI维护预警接口 → Docker容器化 → AWS云端部署

## **1. 项目基础信息**

### **1.1 项目名称**

Wind Turbine Predictive Maintenance Platform

### **1.2 项目主题**

Predicting Wind Turbine Power Output and Detecting Maintenance Risks Using SCADA Data

### **1.3 业务背景（面试必背）**

模拟新能源公司或工业物联网团队机器学习工程师真实工作场景，核心业务问题：**根据实时风速、理论功率曲线、风向等多维 SCADA 特征预测风力发电机应该输出多少电，并通过实际输出与预测输出的差异判断风机是否可能处于亚健康状态**。

风力发电机的运行风险：

- **发电效率下降**：风很大，但实际发电量明显低于理论水平，可能意味着叶片结冰、齿轮箱磨损、偏航角异常或传感器故障

- **停机损失**：风机每停机一小时，就少发一小时电，直接造成清洁能源损失和营收损失

- **人工巡检成本高**：如果没有数据预警，工程师只能定期人工巡检，成本高且难以及时发现早期故障

因此，风力发电机预测性维护是工业互联网和清洁能源领域非常典型的机器学习落地场景。它的核心不是“预测一个抽象数字”，而是帮助企业提前发现设备效率衰减，减少停机时间，提升发电收益。

### **1.4 数据集**

https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

Kaggle Wind Turbine SCADA Dataset 公开风力发电机传感器数据集。

核心特点：

1. **数据规模**：约 50,000 条结构化数据，单表 CSV，适合初学者本地运行

2. **字段简单**：只有少量核心字段，变量关系直观，学习压力低

3. **可视化清晰**：风速、理论功率曲线、风向与实际发电功率之间存在直观关系，非常适合讲解多元线性回归、特征工程和残差分析

核心字段：

1. **Wind Speed (m/s)**：实时风速

2. **LV ActivePower (kW)**：风机实际输出电功率

3. **Theoretical_Power_Curve (KWh)**：该风速下理论上应该达到的电功率

4. **Wind Direction (deg)**：风向角度

## **2. 分阶段学习与开发指南**

### **阶段1：业务理解**

#### **学习目标**

彻底理解预测性维护的业务逻辑，摆脱“只会调包”的入门状态，能独立向面试官讲清楚：为什么一个多特征回归模型也能解决真实工业问题。

#### **必须掌握的核心问题（面试高频）**

##### 什么是 Predictive Maintenance（预测性维护）？

想象一下，你负责管理一片风力发电场。

每台风机都很贵，叶片、齿轮箱、发电机、传感器任何一个部件出问题，都可能导致整台机器效率下降甚至停机。

传统维护方式通常有两种：

1. **坏了再修**：等风机停机后再派工程师去修，损失已经发生

2. **定期巡检**：每隔一段时间人工检查一次，但可能错过早期故障，也可能浪费人力检查健康设备

Predictive Maintenance（预测性维护）就是用传感器数据和机器学习模型提前发现异常信号。

在本项目里，我们把复杂的工业维护问题简化成一个大二学生能听懂的“抓小偷”任务：

如果今天风速很大、理论功率曲线也显示应该高发电，模型预测风机应该发出 **1000 kW** 的电，但实际只发了 **200 kW**，这就很可疑。风没有问题，环境条件也支持发电，但机器没有发出来，说明风机可能正在“带病工作”。

这个差值叫做 **Residual（残差）**：

```text
Residual = Predicted Power - Actual Power
```

当残差长期过大，就可以触发维护预警：

```text
Warning: Maintenance Required
```

一句话总结：预测性维护就是用机器学习模型判断设备“在当前条件下本来应该表现多好”，再把“应该表现”和“实际表现”做对比，从而提前发现设备异常。

##### 为什么本项目适合从 Multiple Linear Regression 开始？

因为风力发电不是只由一个变量决定的。风速当然是最重要因素，但理论功率曲线、风向、风速的非线性变化也会影响发电表现。

风越小，风机几乎不发电；风逐渐变大，发电功率也逐渐升高；达到额定功率后，输出会趋于稳定。

对零基础学生来说，这个关系非常适合用来理解机器学习最基础的思想。先从单变量公式开始：

```text
y = w x + b
```

再自然升级到多特征版本：

```text
y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + ... + b
```

在本项目里：

- y 是 LV ActivePower（实际发电功率）

- x1 是 Wind Speed（风速）

- x2 是 Theoretical_Power_Curve（理论功率）

- x3 是 Wind_Sin（风向正弦特征）

- x4 是 Wind_Cos（风向余弦特征）

- x5 是 Wind_Speed_Squared（风速平方，用来表达简单非线性）

- w1 到 w5 表示每个特征对发电量预测的影响权重

- b 表示基础偏移量

学生依然可以用 Scikit-learn 的 `LinearRegression()`，但训练的是**多元线性回归**。这样既不增加太多学习门槛，又能避免项目看起来只有一个输入特征、工程深度不足。

##### 为什么要做残差分析？

单纯预测发电量还不够，预测性维护的关键是判断设备是否异常。

模型会学习“正常情况下，在当前风速、理论功率曲线和风向条件下，风机应该发多少电”。如果实际发电量明显低于模型预测值，就说明设备表现不符合正常规律。

例子：

```text
当前风速：12 m/s
模型预测发电量：950 kW
实际发电量：400 kW
残差：550 kW
发电衰减比例：57.9%
```

如果我们设定阈值：发电衰减超过 50% 就报警，那么系统会返回：

```json
{
  "status": "Warning: Maintenance Required"
}
```

这就是从机器学习模型走向工业业务闭环的关键一步。

##### 风电公司如何落地使用预测性维护结果？

可以拆解为四个典型自动化场景：

1. **实时异常预警**

风机每隔几分钟上传一次 SCADA 数据。系统把当前风速和实际发电量传给模型，模型判断实际输出是否显著低于预测输出。如果异常，立即生成维护告警。

2. **工程师巡检优先级排序**

一个风电场可能有几十台甚至上百台风机。系统可以按照异常残差大小排序，把最可能出问题的风机排在巡检列表最前面，让工程师优先处理风险最高的设备。

3. **减少停机时间**

如果能在齿轮箱彻底损坏前发现效率下降，企业就可以安排低风速时段进行维护，避免高风速高收益时段突然停机。

4. **提升清洁能源收益**

风机少停机、少低效运行，就能发出更多清洁电力。对新能源公司来说，这既是经济收益，也是 ESG 和可持续发展价值。

#### **学习资料**

Kaggle Wind Turbine SCADA Dataset

https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

Scikit-learn Linear Regression 官方文档

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Scikit-learn Regression Metrics 官方文档

https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

FastAPI 官方文档

https://fastapi.tiangolo.com/

#### **阶段输出**

项目第一版 README.md（包含业务背景、项目价值、数据集介绍、技术路线、维护预警业务逻辑）

### **阶段2：EDA 探索性数据分析**

#### **学习目标**

不做无效画图，通过数据分析理解风速、实际功率、理论功率、风向之间的关系，为后续多特征回归建模和异常检测提供依据。

#### **核心分析任务**

- 风速分布分析：不同风速区间出现频率

- 实际功率分布：风机发电量整体范围与异常低值

- 风速-实际功率散点图：观察风速越大、发电越多的回归关系

- 理论功率 vs 实际功率：观察哪些样本明显低于理论发电曲线

- 风向分析：探索不同风向下发电效率是否存在差异

- 多特征相关性热力图：观察 Wind Speed、Theoretical Power、Wind_Sin、Wind_Cos 与实际功率的相关性

#### **技术工具**

pandas、matplotlib、seaborn、plotly（交互式可视化加分）

#### **阶段输出**

完整 EDA 分析 Jupyter Notebook + 结构化业务洞察总结

### **阶段3：特征工程（新手友好版项目核心）**

#### **学习目标**

掌握入门级工业数据特征构造方法，理解“原始传感器字段”如何转化为“模型可用特征”和“业务可解释指标”。

#### **核心特征体系**

1. **基础传感器特征**

核心逻辑：直接使用 SCADA 数据中最重要的物理变量。

构造：

- Wind Speed (m/s)

- Wind Direction (deg)

- Theoretical_Power_Curve (KWh)

2. **模型输入特征 Model Input Features**

核心逻辑：训练模型时使用多个可提前获得或可由传感器直接计算的 X 特征，预测实际发电功率 Y。

推荐输入特征：

```text
X = [
  Wind Speed (m/s),
  Theoretical_Power_Curve (KWh),
  Wind_Sin,
  Wind_Cos,
  Wind_Speed_Squared
]

Y = LV ActivePower (kW)
```

解释：`LV ActivePower` 是模型要预测的目标，不能作为 X 输入，否则会发生数据泄漏。`Power_Efficiency` 和 `Residual_Ratio` 更适合用于预测之后的维护诊断。

3. **风速非线性特征 Wind Speed Squared**

核心逻辑：风速和功率并不是完全直线关系，加入一个简单平方项，可以让线性模型具备一点点非线性表达能力。

构造：

```text
Wind_Speed_Squared = Wind Speed * Wind Speed
```

解释：这仍然可以用 `LinearRegression()` 训练，但模型已经不再只是看一个风速字段，而是在多个工程特征上学习发电规律。

4. **风向三角函数特征**

核心逻辑：风向是 0 到 360 度的环形变量，不能简单当作普通数字处理。

构造：

```text
Wind_Cos = cos(Wind Direction)
Wind_Sin = sin(Wind Direction)
```

解释：通过三角函数把角度转化为机器学习模型更容易理解的连续数值。

5. **功率转化率特征 Power Efficiency**

核心逻辑：衡量实际发电量与理论发电量之间的差距。

构造：

```text
Power_Efficiency = LV ActivePower / (Theoretical_Power_Curve + epsilon)
```

解释：如果某个样本的转化率长期低于 0.6，说明风机实际输出明显低于理论水平，可能需要维护。

6. **残差特征 Residual**

核心逻辑：用预测值与实际值之间的差距判断设备异常。

构造：

```text
Residual = Predicted Power - Actual Power
Residual_Ratio = Residual / (Predicted Power + epsilon)
```

解释：残差越大，说明风机“应该发的电”和“实际发的电”差距越大，维护风险越高。

#### **阶段输出**

标准化特征工程脚本、特征说明文档、可复用的数据预处理函数

### **阶段4：Multiple Linear Regression 基线模型搭建**

#### **学习目标**

用多元线性回归建立第一个可解释基准，理解训练集、测试集、多个 X 特征、模型拟合、预测、评估指标等核心概念。

#### **模型选择**

- Multiple Linear Regression 多元线性回归

#### **训练目标**

输入多个传感器与工程特征，预测风机实际输出功率：

```text
Input X:
- Wind Speed (m/s)
- Theoretical_Power_Curve (KWh)
- Wind_Sin
- Wind_Cos
- Wind_Speed_Squared

Output: LV ActivePower
```

#### **评估指标（面试必背公式）**

- MAE（平均绝对误差）

- RMSE（均方根误差）

- R2 Score（模型解释方差能力）

#### **阶段输出**

多元线性回归训练代码、模型初步评估报告、MAE/RMSE/R2 分数、特征系数解释

### **阶段5：残差分析与异常检测**

#### **学习目标**

把回归模型转化成预测性维护系统，掌握工业项目中“模型输出如何变成业务报警”的核心思路。

#### **核心开发内容**

- 使用训练好的多元线性回归模型预测 Expected Power

- 计算 Predicted Power 与 Actual Power 的差值

- 设定维护报警阈值

- 输出设备状态：Normal / Warning

#### **报警逻辑示例**

```text
if Residual_Ratio > 0.5:
    status = "Warning: Maintenance Required"
else:
    status = "Normal"
```

#### **阶段输出**

残差分析代码、异常样本可视化、维护报警规则说明文档

### **阶段6：Random Forest / XGBoost 模型优化与Benchmark**

#### **学习目标**

在不增加过高学习负担的前提下，理解工业机器学习项目的模型迭代思想：先用 Multiple Linear Regression 建立可解释基线，再使用 Random Forest 和 XGBoost 捕捉更复杂的非线性关系，并用统一指标做 Benchmark 对比。

#### **核心模型对比**

- Multiple Linear Regression：作为 baseline，优点是简单、可解释、训练快

- Random Forest Regressor：通过多棵决策树学习非线性关系，对异常值和复杂特征组合更鲁棒

- XGBoost Regressor：工业界常用梯度提升树模型，适合结构化表格数据，通常能取得更高预测精度

#### **核心对比维度**

- 预测准确率（MAE/RMSE）

- R2 Score（模型解释能力）

- 可解释性

- 训练速度

- 调参复杂度

- 是否适合最终部署到 FastAPI 服务

#### **阶段输出**

Random Forest / XGBoost 训练代码、模型 Benchmark 对比表格、最终模型选择说明

### **阶段7：模型可解释性与业务解释**

#### **学习目标**

能够用非技术语言解释模型为什么判断某台风机需要维护，形成面试中非常重要的业务表达能力。

#### **核心任务**

- 解释多元线性回归系数：风速、理论功率、风向特征分别如何影响预测功率

- 分析异常样本：为什么当前风速下实际发电量明显偏低

- 对比理论功率曲线：实际功率与理论功率差距有多大

- 总结影响因素：风速、理论功率、风向三角特征、功率转化率、残差比例对维护判断的影响

#### **阶段输出**

模型解释报告、异常案例分析、面试口述版业务解释

### **阶段8：FastAPI 维护预警接口开发**

#### **学习目标**

将离线模型转化为可在线调用的工业服务，体现软件工程和机器学习工程能力。

#### **核心开发内容**

- 基于 FastAPI 搭建后端服务

- 加载训练好的多元线性回归模型

- 开发核心接口：POST /predict-maintenance

#### **接口规范**

请求入参（JSON）：

```json
{
  "wind_speed": 12.0,
  "actual_power": 400.0,
  "wind_direction": 180.0,
  "theoretical_power_curve": 950.0
}
```

响应出参（JSON）：

```json
{
  "predicted_power": 950.0,
  "actual_power": 400.0,
  "input_features": {
    "wind_speed": 12.0,
    "theoretical_power_curve": 950.0,
    "wind_sin": 0.0,
    "wind_cos": -1.0,
    "wind_speed_squared": 144.0
  },
  "residual": 550.0,
  "residual_ratio": 0.579,
  "power_efficiency": 0.421,
  "status": "Warning: Maintenance Required"
}
```

#### **阶段输出**

可本地调试的完整维护预警 API 服务

### **阶段9：Docker 容器化**

#### **学习目标**

掌握机器学习项目工程化打包方式，实现项目环境隔离、一键部署。

#### **核心任务**

- 编写标准化 Dockerfile

- 打包 FastAPI 维护预警服务为镜像

- 实现容器内稳定运行预测服务

#### **阶段输出**

可直接运行的 Docker 镜像、完整容器化部署脚本

### **阶段10：云端部署（项目闭环）**

#### **学习目标**

完成端到端云端落地，实现公网可访问的工业机器学习在线服务。

#### **部署选项（二选一）**

- AWS EC2（优先推荐，适配 AWS 面试）

- Azure App Service（适配微软面试）

#### **部署架构**

FastAPI 服务 + Docker 容器 + 云服务器

#### **阶段输出**

公网可访问的风机维护预警服务

## **3. 模型评估方法（分类 vs 回归）**

### **3.1 为什么必须单独学习模型评估？**

模型训练完成后，不能只说“模型效果不错”，必须用标准指标证明它到底好在哪里、差在哪里。面试官通常会追问两个问题：

- 你的任务是分类还是回归？

- 你为什么选择这个评估指标？

本项目的主任务是**回归问题**：根据风速、理论功率曲线、风向等特征预测风机实际发电功率。但项目中的维护预警也可以自然转化成**分类问题**：例如判断当前风机状态是 `Normal` 还是 `Warning: Maintenance Required`。

### **3.2 回归任务评估指标（本项目重点）**

回归任务预测的是连续数值，适合用于发电功率、销量、价格、温度、收入等场景。

常用指标：

1. **MAE（Mean Absolute Error，平均绝对误差）**

```text
MAE = average(|y_true - y_pred|)
```

解释：预测功率和真实功率平均差多少 kW。优点是直观，适合向非技术面试官解释。

2. **RMSE（Root Mean Squared Error，均方根误差）**

```text
RMSE = sqrt(average((y_true - y_pred)^2))
```

解释：对大误差惩罚更重。如果某些时刻预测功率严重偏离真实功率，RMSE 会明显变高，适合设备异常检测场景。

3. **MAPE（Mean Absolute Percentage Error，平均绝对百分比误差）**

```text
MAPE = average(|y_true - y_pred| / y_true)
```

解释：用百分比衡量误差，例如平均预测偏差 10%。注意：当真实功率为 0 或非常小时，MAPE 会不稳定。

4. **R2 Score（决定系数）**

```text
R2 越接近 1，说明模型解释能力越强
```

解释：衡量模型能解释多少发电功率变化。它适合做模型对比，但不如 MAE/RMSE 直观。

本项目推荐汇报方式：

|Metric|用途|面试解释|
|---|---|---|
|MAE|业务直观误差|平均预测错多少 kW|
|RMSE|惩罚大误差|突出严重预测偏差|
|MAPE|百分比误差|便于比较不同功率区间|
|R2|模型解释能力|衡量模型整体拟合程度|

### **3.3 分类任务评估指标（维护预警扩展必懂）**

分类任务预测的是类别标签，适合用于是否故障、是否欺诈、是否流失、是否需要维护等场景。

在本项目中，可以把残差阈值转化成维护状态标签：

```text
1 = Warning: Maintenance Required
0 = Normal
```

常用指标：

1. **Accuracy（准确率）**

```text
Accuracy = 预测正确样本数 / 总样本数
```

解释：整体预测对了多少。但如果大多数时间风机都是正常状态，Accuracy 可能虚高。

2. **Precision（精确率）**

```text
Precision = TP / (TP + FP)
```

解释：模型判断“需要维护”的样本里，真的异常的比例。适合控制误报，避免工程师被无效告警打扰。

3. **Recall（召回率）**

```text
Recall = TP / (TP + FN)
```

解释：所有真正异常的样本里，模型成功抓住了多少。预测性维护通常更重视 Recall，因为漏报可能导致设备损坏或停机。

4. **F1 Score**

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

解释：Precision 和 Recall 的综合平衡，适合异常样本较少的场景。

5. **ROC-AUC**

解释：衡量模型区分正常与异常样本的能力，越接近 1 越好。适合评估二分类模型整体排序能力。

分类指标选择建议：

|业务目标|优先指标|原因|
|---|---|---|
|不想误报太多|Precision|减少无效巡检|
|不想漏掉故障|Recall|尽量抓住真实异常|
|异常样本较少|F1 / ROC-AUC|比 Accuracy 更可靠|

### **3.4 本项目面试回答模板**

本项目核心是发电功率预测，所以我主要使用 MAE、RMSE、MAPE 和 R2 评估回归模型。MAE 方便解释平均功率误差，RMSE 会更重视大偏差，适合预测性维护场景，因为严重低估或高估功率都可能影响异常判断。除此之外，我会把残差阈值转化成 Normal / Warning 标签，并用 Precision、Recall、F1 和 ROC-AUC 评估维护预警效果。

## **4. 标准化项目目录结构（严格遵守）**

```text
wind-turbine-predictive-maintenance/
|
├── data/                 # 数据集存放目录
|
├── notebooks/            # EDA、实验分析Notebook
|
├── src/                  # 核心源码
│   ├── feature_engineering.py   # 特征工程脚本
│   ├── train.py                 # 模型训练脚本
│   ├── predict.py               # 预测推理脚本
│   └── anomaly_detection.py     # 残差分析与维护预警逻辑
|
├── api/                  # 接口服务代码
│   └── main.py           # FastAPI 主服务
|
├── models/               # 保存训练好的模型文件
|
├── reports/              # EDA报告、模型评估、异常分析
|
├── Dockerfile            # 容器化配置文件
|
├── requirements.txt      # 项目依赖清单
|
└── README.md             # 项目总文档
```

## **5. GitHub 最终展示标准（求职导向）**

### **5.1 核心模块展示**

- **Problem**：清晰阐述风力发电机预测性维护业务问题与价值

- **Dataset**：介绍 Wind Turbine SCADA 数据集规模、字段、适用场景

- **EDA**：展示风速-功率散点图、理论功率与实际功率对比图、异常样本分析

- **Architecture**：完整链路展示

```text
SCADA Data → Multi-feature Engineering → Multiple Linear Regression / Random Forest / XGBoost → Residual Detection → Maintenance Alert API → AWS Cloud Deployment
```

- **Results**：模型效果与异常检测结果

|Model|MAE|RMSE|R2|
|---|---|---|---|
|Multiple Linear Regression|待填充实验结果|待填充实验结果|待填充实验结果|
|Random Forest Regressor|待填充实验结果|待填充实验结果|待填充实验结果|
|XGBoost Regressor|待填充实验结果|待填充实验结果|待填充实验结果|

- **Demo**：公网在线 API 调用示例、维护预警演示

## **6. 面试考点全覆盖清单**

本项目完整覆盖 ML / SDE / Data 实习面试的核心入门考点：

|面试主题|覆盖情况|
|---|---|
|Python 工程开发|OK|
|Pandas 数据处理|OK|
|Matplotlib / Seaborn 可视化|OK|
|多元线性回归基础原理|OK|
|Random Forest / XGBoost 模型对比|OK|
|多特征工程与数据泄漏意识|OK|
|训练集与测试集划分|OK|
|MAE / RMSE / R2 模型评估|OK|
|分类与回归评估指标区分|OK|
|残差分析与异常检测|OK|
|工业物联网业务理解|OK|
|机器学习 API 服务开发|OK|
|Docker 容器化工程|OK|
|AWS 云端部署|OK|
|端到端 ML 系统设计|OK|

## **7. 项目核心优势总结（面试口述版）**

This is an end-to-end Industrial IoT predictive maintenance project for wind turbines. I used SCADA sensor data to build a multiple linear regression model that predicts expected power output from wind speed, theoretical power curve, wind direction sine/cosine features, and engineered wind-speed features. I then applied residual-based anomaly detection to identify turbine underperformance. Instead of only training an offline model, I wrapped the maintenance logic into a FastAPI service, containerized it with Docker, and deployed it on AWS. The project demonstrates both fundamental machine learning understanding and practical ML engineering ability in a clean energy scenario.
