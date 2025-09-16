# CS Programs ChromaDB 项目

一个基于 ChromaDB 和 Google Gemini Embedding 的计算机科学研究生项目智能检索系统。

## 🙏 致谢

本项目基于 [Global CS Application](https://github.com/Global-CS-application/global-cs-application.github.io) 项目的数据构建。感谢该项目为CS申请者提供的宝贵信息和资源。

- **数据来源**: [Global CS Application 项目](https://github.com/Global-CS-application/global-cs-application.github.io)
- **项目分级**: 采用该项目的T0、T1、T2分级标准
- **字段结构**: 详细字段说明请参考原项目文档

## 📁 项目结构

```
cmdb/
├── global_cs_programs.csv          # CS项目数据源
├── cs_programs_notebook.ipynb      # Jupyter notebook主文件
├── cs_programs_notebook.py         # Python脚本版本
├── vectors/                        # ChromaDB持久化存储目录
└── README.md                       # 项目文档
```

## 🔧 环境配置

### 系统要求

- Python 3.8+
- Windows/macOS/Linux

### 依赖安装

```bash
# 安装核心依赖
pip install pandas polars chromadb google-generativeai

# 如果使用Jupyter
pip install jupyter notebook
```

### API 配置

需要获取 Google Gemini API Key：

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建 API Key
3. 在代码中替换 `api_key="YOUR_API_KEY"`

## 📊 数据说明

### 数据来源和结构

- **数据文件**: `global_cs_programs.csv`
- **数据来源**: [Global CS Application 项目](https://github.com/Global-CS-application/global-cs-application.github.io)
- **字段详情**: 请参考原项目的数据结构说明
- **分级标准**: 采用原项目的T0（顶级）、T1（优秀）、T2（良好）分级体系

## 🚀 使用方式

### 1. 启动 Jupyter Notebook

```bash
cd cmdb
jupyter notebook cs_programs_notebook.ipynb
```

### 2. 或运行 Python 脚本

```bash
python cs_programs_notebook.py
```

### 3. 基本工作流程

1. **数据加载**：读取 CSV 文件并预处理
2. **ChromaDB 初始化**：创建持久化向量数据库
3. **文档处理**：将项目信息转换为语义搜索友好的文档
4. **批量添加**：将数据添加到 ChromaDB 集合中
5. **智能查询**：使用自然语言进行语义搜索

## 🔍 查询示例

### 语义搜索 + 过滤器组合

```python
# 查找适合陆本背景的顶级项目
results = collection.query(
    query_texts=["陆本背景申请 录取案例 友好"],
    n_results=5,
    where={"$and": [
        {"tier": {"$in": ["T0", "T1"]}},
        {"admission_data_count": {"$gt": 0}}
    ]}
)
```

### 常用查询模式

#### 1. 按学制筛选

```python
# 短学制项目
collection.query(
    query_texts=["学制短 快速毕业"],
    where={"duration": {"$in": ["9个月", "10个月", "11个月", "12个月", "1年", "一年"]}}
)
```

#### 2. 按地区和等级筛选

```python
# 英国T0项目
collection.query(
    query_texts=["计算机科学 顶级项目"],
    where={"$and": [{"region": "英国"}, {"tier": "T0"}]}
)
```

#### 3. 按论文要求筛选

```python
# 不需要论文的项目
collection.query(
    query_texts=["授课型 coursework"],
    where={"thesis_required": False}
)
```

#### 4. 奖学金机会

```python
# 有奖学金的项目
collection.query(
    query_texts=["奖学金 资助 经济支持"],
    where={"tier": {"$in": ["T0", "T1", "T2"]}}
)
```

## 📈 功能特性

### 🎯 智能搜索

- **语义理解**：支持自然语言查询
- **精确过滤**：基于元数据的多条件筛选
- **相似度排序**：返回最相关的项目

### 📊 数据分析

- 按地区、等级、学制的项目分布统计
- 录取案例完整性分析
- 论文/实习要求统计

### 🔄 持久化存储

- 向量数据本地持久化
- 避免重复 API 调用
- 快速启动和查询

## 🛠️ 高级配置

### 批处理设置

```python
batch_size = 10  # 调整批处理大小
```

### 向量存储路径

```python
chroma_client = chromadb.PersistentClient(path="vectors")
```

### 查询结果数量

```python
n_results = 5  # 调整返回结果数量
```

## 📝 使用技巧

### 1. 查询优化

- 使用具体的关键词而非泛泛的描述
- 结合过滤器缩小搜索范围
- 利用元数据进行精确筛选

### 2. 常见查询场景

- **选校定位**：按等级和地区筛选
- **时间规划**：按学制长短筛选
- **经济考量**：搜索奖学金信息
- **申请策略**：查看录取案例和偏好

### 3. 结果解读

- `相似度分数`：越接近1越相关
- `录取案例数`：数据完整性指标
- `项目等级`：T0 > T1 > T2

## ⚠️ 注意事项

1. **API 限制**：Google Gemini 有免费额度限制
2. **数据更新**：需要手动更新 CSV 文件
3. **存储空间**：向量数据会占用本地存储
4. **网络连接**：首次运行需要网络连接生成向量

## 🔧 故障排除

### 常见问题

**Q: 出现 "ResourceExhausted" 错误**
A: API 配额用完，等待重置或升级账户

**Q: 查询结果不准确**
A: 尝试调整查询关键词或添加过滤条件

**Q: 向量数据库损坏**
A: 删除 `vectors` 目录重新生成

## 📞 技术支持

如遇到问题，请检查：

1. Python 版本和依赖是否正确安装
2. API Key 是否有效
3. CSV 文件格式是否正确
4. 网络连接是否正常

---

**项目版本**: 1.0
**最后更新**: 2025-09-16
