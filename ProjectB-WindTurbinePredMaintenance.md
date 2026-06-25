# **Project B**

**Wind Turbine Predictive Maintenance Platform - Student Guide**

## **0. 项目前言（必读）**

### **0.1 项目定位（核心区别于普通入门项目）**

本项目面向**大二、零机器学习基础**学生，选择风力发电机预测性维护作为工业场景，使用 Kaggle 经典 **Wind Turbine SCADA Dataset** 完成一套从数据分析、线性回归建模、异常检测到 API 部署的端到端机器学习工程项目。

核心逻辑：本项目不追求复杂模型和竞赛排名，而是用一个非常直观的工业问题训练完整工程能力：**业务问题 → 数据理解 → 可视化分析 → 简单特征工程 → 线性回归建模 → 残差异常检测 → API开发 → Docker容器化 → 云端部署 → 系统设计表达**。

### **0.2 项目最终简历亮点**

完成本项目后，简历可直接写入以下成果：

- Developed an Industrial IoT predictive maintenance system for wind turbines using SCADA sensor data

- Built a linear regression asset-performance model to forecast expected power output from wind speed

- Implemented residual-based anomaly detection to identify turbine underperformance and potential mechanical degradation

- Engineered interpretable maintenance features such as power efficiency and wind direction cosine

- Developed a maintenance alert API with FastAPI

- Containerized the service with Docker and deployed it on AWS EC2

### **0.3 适配人群与求职价值**

适配：大二在读、零机器学习基础、准备 ML Engineer / SDE / Data Engineer 暑期实习学生。

优势：相比泰坦尼克号、鸢尾花、房价预测等传统入门 Demo，本项目具备真实工业场景：风力发电机、SCADA 传感器、设备亚健康、预测性维护、清洁能源。它既足够简单，适合第一阶段学习线性回归，又能扩展到 FastAPI、Docker、AWS，形成一套完整的工程化项目闭环。

### **0.4 整体技术路线**

Dataset → EDA业务数据分析 → 新手特征工程 → Linear Regression基线模型 → 残差分析异常检测 → 模型评估 → FastAPI维护预警接口 → Docker容器化 → AWS云端部署

## **1. 项目基础信息**

### **1.1 项目名称**

Wind Turbine Predictive Maintenance Platform

### **1.2 项目主题**

Predicting Wind Turbine Power Output and Detecting Maintenance Risks Using SCADA Data

### **1.3 业务背景（面试必背）**

模拟新能源公司或工业物联网团队机器学习工程师真实工作场景，核心业务问题：**根据实时风速预测风力发电机应该输出多少电，并通过实际输出与预测输出的差异判断风机是否可能处于亚健康状态**。

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

3. **可视化清晰**：风速与发电功率之间存在肉眼可见的强相关关系，非常适合讲解线性回归和残差分析

核心字段：

1. **Wind Speed (m/s)**：实时风速

2. **LV ActivePower (kW)**：风机实际输出电功率

3. **Theoretical_Power_Curve (KWh)**：该风速下理论上应该达到的电功率

4. **Wind Direction (deg)**：风向角度

## **2. 分阶段学习与开发指南**

### **阶段1：业务理解**

#### **学习目标**

彻底理解预测性维护的业务逻辑，摆脱“只会调包”的入门状态，能独立向面试官讲清楚：为什么一个简单线性回归模型也能解决真实工业问题。

#### **必须掌握的核心问题（面试高频）**

##### 什么是 Predictive Maintenance（预测性维护）？

想象一下，你负责管理一片风力发电场。

每台风机都很贵，叶片、齿轮箱、发电机、传感器任何一个部件出问题，都可能导致整台机器效率下降甚至停机。

传统维护方式通常有两种：

1. **坏了再修**：等风机停机后再派工程师去修，损失已经发生

2. **定期巡检**：每隔一段时间人工检查一次，但可能错过早期故障，也可能浪费人力检查健康设备

Predictive Maintenance（预测性维护）就是用传感器数据和机器学习模型提前发现异常信号。

在本项目里，我们把复杂的工业维护问题简化成一个大二学生能听懂的“抓小偷”任务：

如果今天风速很大，模型预测风机应该发出 **1000 kW** 的电，但实际只发了 **200 kW**，这就很可疑。风没有问题，理论上应该能发电，但机器没有发出来，说明风机可能正在“带病工作”。

这个差值叫做 **Residual（残差）**：

```text
Residual = Predicted Power - Actual Power
```

当残差长期过大，就可以触发维护预警：

```text
Warning: Maintenance Required
```

一句话总结：预测性维护就是用机器学习模型判断设备“在当前条件下本来应该表现多好”，再把“应该表现”和“实际表现”做对比，从而提前发现设备异常。

##### 为什么本项目适合从 Linear Regression 开始？

因为风速和风力发电功率之间存在非常直观的正相关关系。

风越小，风机几乎不发电；风逐渐变大，发电功率也逐渐升高；达到额定功率后，输出会趋于稳定。

对零基础学生来说，这个关系非常适合用来理解机器学习最基础的思想：

```text
y = w x + b
```

在本项目里：

- x 是 Wind Speed（风速）

- y 是 LV ActivePower（实际发电功率）

- w 表示风速变化对发电量的影响程度

- b 表示基础偏移量

学生可以先用 `LinearRegression()` 建立第一个模型，再通过散点图、预测线、误差指标理解模型到底在做什么。

##### 为什么要做残差分析？

单纯预测发电量还不够，预测性维护的关键是判断设备是否异常。

模型会学习“正常情况下，某个风速应该发多少电”。如果实际发电量明显低于模型预测值，就说明设备表现不符合正常规律。

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

不做无效画图，通过数据分析理解风速、实际功率、理论功率、风向之间的关系，为后续线性回归建模和异常检测提供依据。

#### **核心分析任务**

- 风速分布分析：不同风速区间出现频率

- 实际功率分布：风机发电量整体范围与异常低值

- 风速-实际功率散点图：观察风速越大、发电越多的回归关系

- 理论功率 vs 实际功率：观察哪些样本明显低于理论发电曲线

- 风向分析：探索不同风向下发电效率是否存在差异

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

2. **功率转化率特征 Power Efficiency**

核心逻辑：衡量实际发电量与理论发电量之间的差距。

构造：

```text
Power_Efficiency = LV ActivePower / (Theoretical_Power_Curve + epsilon)
```

解释：如果某个样本的转化率长期低于 0.6，说明风机实际输出明显低于理论水平，可能需要维护。

3. **风向三角函数特征**

核心逻辑：风向是 0 到 360 度的环形变量，不能简单当作普通数字处理。

构造：

```text
Wind_Cos = cos(Wind Direction)
Wind_Sin = sin(Wind Direction)
```

解释：通过三角函数把角度转化为机器学习模型更容易理解的连续数值。

4. **残差特征 Residual**

核心逻辑：用预测值与实际值之间的差距判断设备异常。

构造：

```text
Residual = Predicted Power - Actual Power
Residual_Ratio = Residual / (Predicted Power + epsilon)
```

解释：残差越大，说明风机“应该发的电”和“实际发的电”差距越大，维护风险越高。

#### **阶段输出**

标准化特征工程脚本、特征说明文档、可复用的数据预处理函数

### **阶段4：Baseline 基线模型搭建**

#### **学习目标**

用最简单的机器学习模型建立第一个可解释基准，理解训练集、测试集、模型拟合、预测、评估指标等核心概念。

#### **模型选择**

- Linear Regression 线性回归

#### **训练目标**

输入风速等传感器特征，预测风机实际输出功率：

```text
Input: Wind Speed
Output: LV ActivePower
```

#### **评估指标（面试必背公式）**

- MAE（平均绝对误差）

- RMSE（均方根误差）

- R2 Score（模型解释方差能力）

#### **阶段输出**

线性回归训练代码、模型初步评估报告、MAE/RMSE/R2 分数

### **阶段5：残差分析与异常检测**

#### **学习目标**

把回归模型转化成预测性维护系统，掌握工业项目中“模型输出如何变成业务报警”的核心思路。

#### **核心开发内容**

- 使用训练好的线性回归模型预测 Expected Power

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

### **阶段6：模型优化与Benchmark**

#### **学习目标**

在不增加过高学习负担的前提下，理解模型迭代思想：先做简单模型，再尝试稍复杂模型进行对比。

#### **可选模型对比**

- Linear Regression

- Polynomial Regression（二次特征，观察风速与功率的非线性关系）

- Random Forest Regressor（作为进阶拓展）

#### **核心对比维度**

- 预测准确率（MAE/RMSE）

- 可解释性

- 训练速度

- 对初学者的理解难度

#### **阶段输出**

模型 Benchmark 对比表格、最终模型选择说明

### **阶段7：模型可解释性与业务解释**

#### **学习目标**

能够用非技术语言解释模型为什么判断某台风机需要维护，形成面试中非常重要的业务表达能力。

#### **核心任务**

- 解释线性回归系数：风速增加时，预测功率如何变化

- 分析异常样本：为什么当前风速下实际发电量明显偏低

- 对比理论功率曲线：实际功率与理论功率差距有多大

- 总结影响因素：风速、风向、理论功率、功率转化率对维护判断的影响

#### **阶段输出**

模型解释报告、异常案例分析、面试口述版业务解释

### **阶段8：FastAPI 维护预警接口开发**

#### **学习目标**

将离线模型转化为可在线调用的工业服务，体现软件工程和机器学习工程能力。

#### **核心开发内容**

- 基于 FastAPI 搭建后端服务

- 加载训练好的线性回归模型

- 开发核心接口：POST /predict-maintenance

#### **接口规范**

请求入参（JSON）：

```json
{
  "wind_speed": 12.0,
  "actual_power": 400.0,
  "wind_direction": 180.0
}
```

响应出参（JSON）：

```json
{
  "predicted_power": 950.0,
  "actual_power": 400.0,
  "residual": 550.0,
  "residual_ratio": 0.579,
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

## **3. 标准化项目目录结构（严格遵守）**

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

## **4. GitHub 最终展示标准（求职导向）**

### **4.1 核心模块展示**

- **Problem**：清晰阐述风力发电机预测性维护业务问题与价值

- **Dataset**：介绍 Wind Turbine SCADA 数据集规模、字段、适用场景

- **EDA**：展示风速-功率散点图、理论功率与实际功率对比图、异常样本分析

- **Architecture**：完整链路展示

```text
SCADA Data → Feature Engineering → Linear Regression → Residual Detection → Maintenance Alert API → AWS Cloud Deployment
```

- **Results**：模型效果与异常检测结果

|Model|MAE|RMSE|R2|
|---|---|---|---|
|Linear Regression|待填充实验结果|待填充实验结果|待填充实验结果|
|Polynomial Regression|待填充实验结果|待填充实验结果|待填充实验结果|
|Random Forest Regressor（可选）|待填充实验结果|待填充实验结果|待填充实验结果|

- **Demo**：公网在线 API 调用示例、维护预警演示

## **5. 面试考点全覆盖清单**

本项目完整覆盖 ML / SDE / Data 实习面试的核心入门考点：

|面试主题|覆盖情况|
|---|---|
|Python 工程开发|OK|
|Pandas 数据处理|OK|
|Matplotlib / Seaborn 可视化|OK|
|线性回归基础原理|OK|
|训练集与测试集划分|OK|
|MAE / RMSE / R2 模型评估|OK|
|残差分析与异常检测|OK|
|工业物联网业务理解|OK|
|机器学习 API 服务开发|OK|
|Docker 容器化工程|OK|
|AWS 云端部署|OK|
|端到端 ML 系统设计|OK|

## **6. 项目核心优势总结（面试口述版）**

This is an end-to-end Industrial IoT predictive maintenance project for wind turbines. I used SCADA sensor data to build a linear regression model that predicts expected power output from wind speed, then applied residual-based anomaly detection to identify turbine underperformance. Instead of only training an offline model, I wrapped the maintenance logic into a FastAPI service, containerized it with Docker, and deployed it on AWS. The project demonstrates both fundamental machine learning understanding and practical ML engineering ability in a clean energy scenario.

