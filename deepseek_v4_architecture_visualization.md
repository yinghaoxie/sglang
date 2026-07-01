# DeepSeek-V4-Flash 架构可视化分析报告

## 📊 模型规模概览

| 指标 | 数值 |
|------|------|
| **总层数** | 43 层 |
| **总参数量** | ~79.8 Billion |
| **模型大小** | 148.65 GB |
| **MoE 专家数** | 256 路由 + 1 共享 |
| **每 token 激活专家** | 6 个 |
| **参数激活率** | ~5.5% |

## ⚡ 三层并行注意力机制

### MLA (Multi-head Latent Attention)
- 核心权重：wkv.weight, q_norm, kv_norm
- KV Cache 压缩 70%
- 低秩分解保持精度

### CSA (Compressed Sparse Attention)  
- 核心权重：attn_sink
- TopK=512 稀疏选择
- 复杂度 O(S²) → O(S log S)

### HCA (Hierarchical Compressed Attention)
- C128 压缩器 (128:1)
- 层次化块选择
- 跨层 HC 融合

## 🔀 MoE 结构
- 256 路由专家 + 1 共享专家
- gate.weight, gate.tid2eid
- 每专家 w1/w2/w3 三线性层

## 🎯 HC 机制
- hc_attn_base/fn/scale
- hc_ffn_base/fn/scale
- Sinkhorn 迭代归一化

*数据来源：ModelScope DeepSeek-V4-Flash*
