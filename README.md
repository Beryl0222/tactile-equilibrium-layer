# 触觉平衡层的隐式梯度与稳定学习提交

这是面向qualified-equilibrium的 Python 数值项目资料。数据全部由本地确定种子生成，不含真实业务或用户记录。

`data/materials.json` 保存张量、身份、参数与容量配置，`data/events.json` 保存有先后关系的操作输入。数学语义、状态含义与数值范围见 [数据约定](docs/data-contract.md)。事件记录不包含计算结果，不能视作已经执行的回执。

使用 Python 3.11，依赖版本记录在 `requirements.txt`。可在独立环境中运行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python inspect_materials.py
.venv/bin/python -m unittest discover -s tests -v
```

读取入口会核对资料结构、数值有限性及引用，打印资料指纹和操作数；这些检查只说明输入可读取，不宣称算子、训练或业务流程已经执行。容量配置属于可生成输入的规格，不是测得的性能。实际算子运行入口及用法由其实现版本提供。
