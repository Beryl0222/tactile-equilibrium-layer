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

较大输入可用 `.venv/bin/python make_capacity.py --output /tmp/product-capacity` 生成到一个不存在的目录。它只创建输入数组和元数据，不运行算子；`arrays.npz` 的键、形状和类型记录在生成目录的 `metadata.json`。同一种子得到相同数组内容。

## 平衡层实现

仓库在 `equilibrium_layer/` 中实现了不依赖平衡网络库或自动微分框架的核心能力：

- 反复求解 `Z = tanh(A Z W + X U + b)`，直到满足公开残差界；
- 用矩阵自由的 `(I-J_Z)^T λ = G` 求解隐式梯度，并返回 `X/W/U/b` VJP；
- 独立检查反向线性残差与逆算子风险；
- 按稳定节点 ID 映射热启动，拒绝把旧数组位置当作新节点事实；
- 按真正合格样本的非负权重聚合共享参数梯度；
- 校验参数代次、图修订/指纹、梯度来源和整族稳定性证明；
- 小图（活跃节点数 `<16`）提供稠密 Jacobian/SVD 与有限差分对照；
- 大图只使用 COO/CSR `A`、`A^T` 和稀疏矩阵乘法，不物化 `N²` 邻接或 `(ND)²` Jacobian。

根目录的 `equilibrium.py`、`qualified_equilibrium.py`、`solution.py`、`api.py` 与 `core.py` 只是导入别名；主 API 位于 `equilibrium_layer`。

### 常用入口

```sh
# 读取并回放 data/events.json，输出每次前向、反向和更新资格
python -m equilibrium_layer

# 小图稠密 oracle 与中心有限差分对照
python finite_difference_check.py

# 生成并运行 2048 节点容量样本
python make_capacity.py --output /tmp/product-capacity
python run_capacity.py --input /tmp/product-capacity --check-backward

# 全部单元测试
python -m unittest discover -s tests -v

# 不依赖 materials.json 的可更换图/参数示例
python examples/custom_graph.py
```

### Python 调用路径

```python
import json
from equilibrium_layer import (
    Graph, Parameters, GraphFamily, EquilibriumSession, process_batch
)

materials = json.load(open("data/materials.json", encoding="utf-8"))
params = Parameters.from_mapping(materials["parameters"], generation=0)
family = GraphFamily.from_mapping(materials["families"][0])
session = EquilibriumSession(params, [family])
window = session.begin_window(window_id="frame-1")

g0 = Graph.from_mapping(materials["graphs"][0])
h0 = session.solve(g0, max_actions=100)       # 自动按 graph_id 使用已接受热启动
print(h0.viewable, h0.residual, h0.accepted_node_ids)

if h0.viewable:
    grad = session.backward(
        h0,
        materials["upstream"][0],             # 也可传 {node_id: margin}
        max_actions=100,
        inverse_risk_limit=100.0,
    )
    if window.add(grad):                      # 只有 eligible 且代次一致才进入分母
        print(grad.gradients.W)

candidate = Parameters.from_mapping(materials["candidate_parameters"][0])
update = session.propose_update(candidate, window, family_id="contact-family")
print(update.accepted, update.status, session.generation)
```

批量样本可用 `process_batch(...)`；单个样本在求解或梯度阶段失败不会中断同批其他样本。

### 三种资格，分开判断

| 问题 | 主要字段 | 含义 |
| --- | --- | --- |
| 能否作为重建查看 | `ForwardReceipt.viewable` / `accepted` | 非有限状态不接受；只有 `max|Z-F(Z)| <= 1e-8 + 1e-6 max|Z|` 才接受。预算耗尽时 `state is None`，`candidate_state` 仅供诊断，也不会提交为下一帧热启动。 |
| 能否进入训练 | `GradientReceipt.trainable` / `eligible` | 前向已接受、隐式转置系统相对残差 `<=1e-6`、逆算子风险在请求上限内，三者同时成立。前向仍可查看，但梯度 receipt 可为拒绝。 |
| 能否接受学习更新 | `UpdateResult.accepted` | 窗口分母为正、梯度来源/参数代次/图修订一致，且候选参数得到整族 `certified` 稳定性证明。拒绝时旧参数、权重分母和已接受热启动均保留。 |

拒绝阶段可从：

- 前向：`forward.status`、`forward.reason`；
- 梯度：`gradient.status`、`gradient.reasons`、`gradient.linear`、`gradient.risk`；
- 窗口归属：`WindowAttribution.reason`；
- 更新：`update.status`、`update.reasons`、`update.stability.status`

查出。常见原因包括 `action_budget_exhausted`、`linear_residual_certificate_unavailable`、`inverse_norm_risk_exceeds_limit`、`parameter_generation_mismatch`、`mixed_graph_revisions`、`stability_violated` 和 `stability_evidence_insufficient`。显式关闭的 forward 默认在申请伴随梯度时抛出 `ClosedForwardError`；需要状态式拒绝回执可传 `raise_if_closed=False`。

### 风险口径

默认 `inverse_risk_limit` 比较谱逆范数的严格证书：

- 小图：稠密 `I-J` 的 SVD，`||(I-J)^{-1}||_2 = 1/sigma_min(I-J)`；
- 大图：由精确的 `||J||_1`、`||J||_∞` 得 `||J||_2 <= sqrt(||J||_1 ||J||_∞)`，再用 Neumann 界 `1/(1-||J||_2)`。

`RiskReport` 还给出 max-row-Euclidean 压缩界、条件数/条件界、`inverse_norm_excess` 与 `condition_excess`。如果请求中的 `inverse_risk_limit` 采用“超过恒等放大的风险”口径，可传 `risk_metric="amplification_excess"`（或 `"condition_excess"`）；报告同时保留原始范数是否满足上限。

### 整族稳定性

对范数 `max_i ||Z_i||_2`，有

```text
Lip(F) <= max_i sum_j |A_ij| * ||W||_2 <= family.max_abs_row_sum * ||W||_2
```

因此 `R * ||W||_2 <= rho` 给出整族 `certified` 证明。若可构造的族成员也超过 `rho`，标记 `violated`；若上界超过但现有权重/度数声明不足以证明违规，则标记 `insufficient_evidence`。缺少族声明同样是证据不足，不会被当成稳定。

### 资源约定

活跃图最多 2048 节点、每节点最多 12 条边、容量状态维 8–32；实现也允许资料中的 3 维直接对照样本，并在 `support_report()["warnings"]` 标明它低于容量优化区间。空有效图返回空状态且不计训练权重。每个前向 receipt 只保留平衡点和 tanh gate 检查点，不保留展开历史；Anderson 历史仅使用很小的固定内存。最大容量 profile 的单个活跃检查点远低于 16 MiB。

替换图时构造新的 `Graph`（或 `Graph.from_mapping`），替换参数时构造新的 `Parameters`；有限差分入口 `finite_difference_gradients` 会对扰动参数重新运行核心求解器，可作为更换图/参数后的数值对照模板。

