# DeepSeek V4 架构可视化 (SGLang 实现)

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DeepSeek V4 模型架构                                   │
│                         (基于 SGLang 实现)                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

输入: input_ids [batch_size, seq_len]
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  第 0 步：Token Embedding                                                      │
│  VocabParallelEmbedding                                                       │
│  形状：[vocab_size=129280, hidden_size=4096] → [batch, hc_mult=4, 4096]       │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                    DeepSeekV4Model (共 43 层)                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Decoder Layer 0 (非哈希层，layer_id >= n_hash_layers=3)              │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  HC Pre (Attention Path)                                      │  │    │
│   │  │  - hc_mult=4, hc_sinkhorn_iters=20                            │  │    │
│   │  │  - 输入：[batch, 4096] → 输出：[batch, 4096]                  │  │    │
│   │  │  - 包含 input_layernorm (RMSNorm)                             │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │            │                                                         │    │
│   │            ▼                                                         │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  MQALayer (Multi-head Query Attention)                        │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  wq_a / wkv (或 fused wqkv_a)                           │  │  │    │
│   │  │  │  - wq_a: [4096, q_lora_rank=1024]                       │  │  │    │
│   │  │  │  - wkv: [4096, head_dim=512]                            │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  q_norm (RMSNorm, q_lora_rank=1024)                     │  │  │    │
│   │  │  │  kv_norm (RMSNorm, head_dim=512)                        │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  wq_b (ColumnParallelLinear)                            │  │  │    │
│   │  │  │  - [1024, n_heads*head_dim=64*512=32768]                │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  Compressor (可选，compress_ratio ∈ {0, 4, 128})         │  │  │    │
│   │  │  │  - 压缩 KV Cache                                        │  │  │    │
│   │  │  │  - 应用 Rotary Embedding (RoPE)                         │  │  │    │
│   │  │  │    * qk_rope_head_dim=64                                │  │  │    │
│   │  │  │    * qk_nope_head_dim=448                               │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  Attention Backend (DeepseekV4AttnBackend)              │  │  │    │
│   │  │  │  - MLA (Multi-head Latent Attention)                    │  │  │    │
│   │  │  │  - kv_lora_rank=512                                     │  │  │    │
│   │  │  │  - 支持 FlashAttention / Triton 后端                      │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  wo_a / wo_b (输出投影)                                  │  │  │    │
│   │  │  │  - wo_a: [n_heads*head_dim//o_groups, o_lora_rank]      │  │  │    │
│   │  │  │  - wo_b: [o_lora_rank=1024, hidden_size=4096]           │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │            │                                                         │    │
│   │            ▼                                                         │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  HC Post (Attention→FFN 过渡)                                   │  │    │
│   │  │  - 融合残差连接                                                │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │            │                                                         │    │
│   │            ▼                                                         │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  HC Pre (FFN Path)                                            │  │    │
│   │  │  - 包含 post_attention_layernorm (RMSNorm)                    │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │            │                                                         │    │
│   │            ▼                                                         │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  DeepseekV2MoE (混合专家网络)                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  MoEGate (路由网络)                                      │  │  │    │
│   │  │  │  - num_experts_per_tok=6                                │  │  │    │
│   │  │  │  - n_routed_experts=256                                 │  │  │    │
│   │  │  │  - topk_method="noaux_tc"                               │  │  │    │
│   │  │  │  - scoring_func="sqrtsoftplus"                          │  │  │    │
│   │  │  │  - n_group=8, topk_group=8                              │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  │            │                                                   │  │    │
│   │  │            ▼                                                   │  │    │
│   │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│   │  │  │  FusedMoE (专家层)                                       │  │  │    │
│   │  │  │  - 256 个路由专家 + 1 个共享专家                           │  │  │    │
│   │  │  │  - moe_intermediate_size=2048                           │  │  │    │
│   │  │  │  - 支持 FP8/FP4 量化                                       │  │  │    │
│   │  │  │  - 激活函数：SiLU                                         │  │  │    │
│   │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │            │                                                         │    │
│   │            ▼                                                         │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │  HC Post (FFN 输出)                                            │  │    │
│   │  │  - 输出：[batch, hc_mult=4, hidden_size=4096]                 │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                              × 43 层                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Decoder Layer 1-2 (哈希层，layer_id < n_hash_layers=3)              │    │
│   │  - 使用 HashTopK 路由                                                │    │
│   │  - 其他结构相同                                                      │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  Final LayerNorm (RMSNorm)                                                    │
│  仅最后一层输出经过此归一化                                                    │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  HC Head Fusion                                                               │
│  - 将 hc_mult=4 维度压缩回单个隐藏状态                                           │
│  - fused_hc_head 核融合 RMSNorm + 线性投影                                      │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  Logits Processor (ParallelLMHead)                                            │
│  - 映射到 vocab_size=129280                                                   │
│  - 可选 FP8 量化                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
输出：logits [batch_size, vocab_size]
```

---

## 关键组件详解

### 1. mHC (multi-Head Coefficient) 机制

DeepSeek V4 的核心创新是 **mHC** 架构，它在每个 Transformer 块的输入和输出都维护一个额外的 `hc_mult=4` 维度：

```
标准 Transformer:          DeepSeek V4 (mHC):
  [batch, hidden]             [batch, hc_mult=4, hidden]
       │                            │
       ▼                            ▼
  ┌─────────┐                  ┌─────────────┐
  │LayerNorm│                  │  HC Pre     │ ← Sinkhorn 迭代归一化
  └─────────┘                  │ (20 次迭代)  │
       │                       └─────────────┘
       ▼                            │
  ┌─────────┐                      ▼
  │Attention│                  ┌─────────┐
  └─────────┘                  │Attention│
       │                       └─────────┘
       ▼                            │
  ┌─────────┐                      ▼
  │LayerNorm│                  ┌─────────────┐
  └─────────┘                  │  HC Post    │ ← 残差融合
       │                       └─────────────┘
       ▼                            │
  ┌─────────┐                      ▼
  │  FFN    │                  ┌─────────────┐
  └─────────┘                  │  HC Pre     │
       │                       └─────────────┘
       ▼                            │
  ┌─────────┐                      ▼
  │  Add    │                  ┌─────────┐
  └─────────┘                  │   MoE     │
       │                       └─────────┘
       ▼                            │
  [batch, hidden]                   ▼
                              [batch, hc_mult, hidden]
```

**HC Pre 操作**:
```python
# 伪代码
def hc_pre(x, hc_fn, hc_scale, hc_base, norm):
    # 1. 计算混合系数 (mixes)
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1) + rms_norm_eps)
    mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
    
    # 2. Sinkhorn 迭代归一化 (20 次)
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base,
        hc_mult=4, 
        sinkhorn_iters=20,
        eps=1e-6
    )
    
    # 3. 加权求和
    y = (pre.squeeze(1).unsqueeze(-1) * x_flat).sum(dim=1)
    return y, post, comb
```

**HC Post 操作**:
```python
def hc_post(x, residual, post, comb):
    # 融合残差和组合系数
    output = (
        post.unsqueeze(-1) * x.unsqueeze(1) +
        (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    )
    return output
```

---

### 2. MQA (Multi-head Query Attention) / MLA (Multi-head Latent Attention)

DeepSeek V4 使用 **MLA** 注意力机制，通过低秩分解减少 KV Cache 大小：

```
输入：hidden_states [batch, hidden_size=4096]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Q 路径                                                    │
│  wq_a: [4096, q_lora_rank=1024]                          │
│  q_norm: RMSNorm(1024)                                    │
│  wq_b: [1024, n_heads*head_dim=64*512=32768]             │
│  输出：Q [batch, 64, 512]                                 │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  KV 路径 (低秩)                                            │
│  wkv: [4096, head_dim=512]                               │
│  kv_norm: RMSNorm(512)                                    │
│  输出：KV_latent [batch, 512]                             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  RoPE (旋转位置编码)                                       │
│  - qk_rope_head_dim=64 (仅对前 64 维应用 RoPE)               │
│  - qk_nope_head_dim=448 (无位置编码)                       │
│  - rope_theta=10000 (或 compress_rope_theta=40000)        │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Compressor (可选，compress_ratio ∈ {0, 4, 128})           │
│  - 当 compress_ratio > 0 时，压缩 KV Cache                   │
│  - 减少显存占用                                             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Attention Backend                                         │
│  - FlashAttention / Triton 实现                             │
│  - 支持 DSA (Dynamic Sparse Attention)                    │
│  - 支持 CP (Context Parallel)                             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  输出投影                                                  │
│  wo_a: [n_heads*head_dim//o_groups, o_lora_rank=1024]    │
│  wo_b: [1024, hidden_size=4096]                          │
└───────────────────────────────────────────────────────────┘
```

**关键参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `num_attention_heads` | 64 | 注意力头数 |
| `head_dim` | 512 | 每头维度 (qk_nope+qk_rope) |
| `qk_nope_head_dim` | 448 | 无位置编码部分 |
| `qk_rope_head_dim` | 64 | 应用 RoPE 部分 |
| `q_lora_rank` | 1024 | Q 的低秩维度 |
| `o_lora_rank` | 1024 | 输出的低秩维度 |
| `kv_lora_rank` | 512 | KV 的低秩维度 |
| `o_groups` | 8 | 输出分组数 |

---

### 3. MoE (Mixture of Experts)

DeepSeek V4 使用稀疏 MoE 架构：

```
输入：hidden_states [batch, hidden_size=4096]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  MoEGate (路由网络)                                        │
│  - 输入：[batch, 4096]                                    │
│  - 输出：topk_indices [batch, 6], topk_weights [batch, 6] │
│                                                           │
│  路由算法：                                                 │
│  1. 计算专家分数：scores = Linear(hidden)                 │
│  2. 分组：按 n_group=8 分组                                 │
│  3. 选择 topk_group=8 组                                    │
│  4. 每组内选 top expert                                     │
│  5. 评分函数：sqrtsoftplus                                 │
│  6. 缩放因子：routed_scaling_factor=1.5                    │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  FusedMoE Kernel                                          │
│  - 256 个路由专家 (n_routed_experts)                        │
│  - 1 个共享专家 (n_shared_experts)                          │
│  - 每 token 激活 6 个专家 (num_experts_per_tok)              │
│                                                           │
│  每个专家结构：                                              │
│  ┌─────────────┐      ┌─────────────┐                     │
│  │ Gate Proj   │ ───► │  SiLU       │                     │
│  │ [4096,2048] │      │  激活       │                     │
│  └─────────────┘      └──────┬──────┘                     │
│                              │                             │
│                     ┌────────▼────────┐                   │
│                     │  Up Proj        │                   │
│                     │  [4096, 2048]   │                   │
│                     └────────┬────────┘                   │
│                              │                             │
│                     ┌────────▼────────┐                   │
│                     │  Down Proj      │                   │
│                     │  [2048, 4096]   │                   │
│                     └─────────────────┘                   │
└───────────────────────────────────────────────────────────┘
        │
        ▼
输出：[batch, hidden_size=4096]
```

**MoE 配置**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `n_routed_experts` | 256 | 路由专家总数 |
| `n_shared_experts` | 1 | 共享专家数 (所有 token 共用) |
| `num_experts_per_tok` | 6 | 每 token 激活专家数 |
| `moe_intermediate_size` | 2048 | 专家中间层维度 |
| `n_group` | 8 | 专家分组数 |
| `topk_group` | 8 | 选择的组数 |
| `scoring_func` | "sqrtsoftplus" | 评分函数 |
| `norm_topk_prob` | True | 是否归一化权重 |

**哈希层 vs 普通层**:
- **前 3 层** (`layer_id < n_hash_layers=3`): 使用 `HashTopK` 路由
  - 基于 token ID 的哈希路由（确定性）
  - 无需计算门控分数
- **后 40 层**: 使用标准 `TopK` 路由
  - 动态门控路由（基于输入内容）
  - `noaux_tc` 方法（无辅助损失）

---

### 4. 分布式策略

DeepSeek V4 支持多种并行策略：

```
┌─────────────────────────────────────────────────────────────────┐
│                    并行策略总览                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TP (Tensor Parallel)                                           │
│  ├── wq_b: ColumnParallelLinear                                 │
│  ├── wo_a: ColumnParallelLinear                                 │
│  └── MoE 专家：专家级 TP 切分                                      │
│                                                                 │
│  EP (Expert Parallel)                                           │
│  ├── ep_size: 专家并行度                                        │
│  ├── 每个 GPU 持有部分专家                                       │
│  └── 需要 All-to-All 通信                                        │
│                                                                 │
│  DP (Data Parallel)                                             │
│  ├── attn_dp_size: 注意力数据并行                               │
│  ├── dp_gather_partial: 部分 gather                             │
│  └── dp_reduce_scatterv: 变量长度 reduce-scatter                │
│                                                                 │
│  CP (Context Parallel)                                          │
│  ├── dsa_enable_prefill_cp: 启用上下文并行                      │
│  ├── cp_split_and_rebuild_data: 数据切分                        │
│  └── dsa_cp_gather/reduce_scatter: 通信原语                     │
│                                                                 │
│  PP (Pipeline Parallel)                                         │
│  ├── pp_group: 流水线并行组                                     │
│  ├── make_layers: 分层分配                                      │
│  └── PPProxyTensors: 跨阶段张量传递                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. 量化支持

DeepSeek V4 支持多种量化格式：

```
┌─────────────────────────────────────────────────────────────────┐
│                      量化配置                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FP8 (F8_E4M3)                                                  │
│  ├── 专家权重：F8_E4M3                                          │
│  ├── 激活：F8_E4M3                                              │
│  └── sglang_per_token_group_quant_fp8                           │
│                                                                 │
│  FP4 (MXFP4)                                                    │
│  ├── 专家权重：U8/I8/F4 打包格式                                 │
│  ├── is_fp4_experts=True                                        │
│  └── 需要特殊解包内核                                            │
│                                                                 │
│  W8A8                                                           │
│  ├── 权重：INT8                                                 │
│  ├── 激活：INT8                                                 │
│  └── DeepGEMM 后端                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 数据流示例

假设输入序列长度为 1024，batch_size=1：

```
Step 0: 输入
  input_ids: [1, 1024]  (token IDs)

Step 1: Token Embedding
  embed_tokens: [1, 1024, 4096]
  → unsqueeze + repeat: [1, 1024, hc_mult=4, 4096]
  → flatten: [1*1024, 4, 4096]

Step 2-N+1: N 个 Decoder Layer
  对于每一层:
  
  2a. HC Pre (Attention)
      输入：[tokens, 4, 4096]
      → hc_pre: [tokens, 4096], post[tokens, 4], comb[tokens, 4, 4]
      
  2b. MQA Attention
      Q: [tokens, 64, 512]
      KV: [tokens, 512] (压缩后)
      → Attention 输出：[tokens, 4096]
      
  2c. HC Post → HC Pre (FFN)
      → hc_post: [tokens, 4, 4096]
      → hc_pre: [tokens, 4096], post, comb
      
  2d. MoE
      Gate: 选择 top-6 专家
      Experts: [tokens, 4096]
      
  2e. HC Post
      → 输出：[tokens, 4, 4096]

Step N+2: Final Norm + HC Head
  norm: RMSNorm([tokens, 4, 4096])
  hc_head: [tokens, 4096]  (压缩 hc_mult 维度)

Step N+3: Logits
  lm_head: [tokens, vocab_size=129280]
```

---

## 性能优化技术

### 1. 跨层融合 (Cross-Layer Fusion)

```python
# 传统方式 (逐层计算)
for layer in layers:
    residual = hidden_states
    hidden_states = hc_pre(...)
    hidden_states = attention(...)
    hidden_states = hc_post(hidden_states, residual, ...)
    
# DeepSeek V4 (跨层融合)
prev_residual, prev_post, prev_comb = None, None, None
for layer in layers:
    if prev_residual is not None:
        # 融合上一层的 HC Post 和当前层的 HC Pre
        hidden_states, residual, post, comb = mhc_fused_post_pre(
            hidden_states, prev_residual, prev_post, prev_comb, ...
        )
    else:
        hidden_states, post, comb, _ = hc_pre(...)
        residual = hidden_states
    
    hidden_states = attention(...)
    prev_residual, prev_post, prev_comb = residual, post, comb

# 最后一层完成最终的 HC Post
hidden_states = hc_head(hidden_states, prev_residual, prev_post, prev_comb)
```

### 2. 多流重叠 (Multi-Stream Overlap)

```python
# 使用多个 CUDA Stream 并行执行
alt_streams = [torch.cuda.Stream() for _ in range(5)]

# Stream 0: HC Pre
with torch.cuda.stream(alt_streams[0]):
    hidden_states, post, comb = hc_pre(...)

# Stream 1: Attention
with torch.cuda.stream(alt_streams[1]):
    q, kv = wq_a(...), wkv(...)

# Stream 2: MoE Gate
with torch.cuda.stream(alt_streams[2]):
    topk_indices, topk_weights = gate(...)

# 同步等待
for stream in alt_streams:
    stream.synchronize()
```

### 3. 内存优化

- **KV Cache 压缩**: `compress_ratio ∈ {0, 4, 128}`
- **共享专家局部计算**: `SGLANG_DP_SHARED_EXPERT_LOCAL`
- **Reduce-Scatter 替代 All-Reduce**: `SGLANG_DP_USE_REDUCE_SCATTER`

---

## 总结

DeepSeek V4 在 SGLang 中的实现具有以下特点：

1. **mHC 架构**: 通过 `hc_mult=4` 维度和 Sinkhorn 迭代实现更稳定的训练
2. **MLA 注意力**: 低秩 KV 分解减少显存占用
3. **稀疏 MoE**: 256 专家 + 6 激活/令牌，提高计算效率
4. **混合路由**: 前 3 层哈希路由 + 后 40 层动态路由
5. **丰富并行**: TP/EP/DP/CP/PP 全方位支持
6. **量化友好**: FP8/FP4/W8A8 多种格式
7. **内核融合**: 跨层 HC Pre/Post 融合，减少内存访问

这种设计使 DeepSeek V4 能够在保持高性能的同时，显著降低训练和推理成本。
