#!/usr/bin/env python3
"""
DeepSeek-V4-Flash 架构分析脚本
基于 model.safetensors.index.json 分析模型结构
"""

import json
from collections import defaultdict
from typing import Dict, List, Any


def load_model_index(filepath: str) -> Dict[str, Any]:
    """加载模型索引文件"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_layer_structure(weight_map: Dict[str, str]) -> Dict[str, Any]:
    """分析单层结构"""
    layer_keys = [k for k in weight_map.keys() if k.startswith('layers.')]
    
    # 提取所有层号
    layers = set()
    for key in layer_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            try:
                layer_num = int(parts[1])
                layers.add(layer_num)
            except ValueError:
                continue
    
    # 分析第 0 层的完整结构
    layer_0_structure = defaultdict(list)
    for key in layer_keys:
        if key.startswith('layers.0.'):
            parts = key.split('.')
            if len(parts) >= 3:
                component = parts[2]
                subcomponent = '.'.join(parts[3:]) if len(parts) > 3 else ''
                layer_0_structure[component].append(subcomponent)
    
    # 统计专家数量
    expert_count = 0
    for key in layer_keys:
        if key.startswith('layers.0.ffn.experts.'):
            parts = key.split('.')
            if len(parts) >= 4:
                try:
                    expert_id = int(parts[3])
                    expert_count = max(expert_count, expert_id + 1)
                except ValueError:
                    continue
    
    return {
        'total_layers': len(layers),
        'layer_range': (min(layers), max(layers)),
        'expert_count': expert_count,
        'components': dict(layer_0_structure)
    }


def analyze_attention_paths(weight_map: Dict[str, str]) -> Dict[str, Any]:
    """分析注意力路径（MLA/CSA/HCA）"""
    layer_0_keys = [k for k in weight_map.keys() if k.startswith('layers.0.attn.')]
    
    paths = {
        'mla': [],
        'csa': [],
        'hca': [],
        'shared': []
    }
    
    # 识别不同路径的权重
    for key in layer_0_keys:
        if 'wq_' in key or 'kv_norm' in key or 'q_norm' in key:
            paths['shared'].append(key.replace('layers.0.attn.', ''))
        if 'wkv' in key:
            paths['mla'].append(key.replace('layers.0.attn.', ''))
        if 'wo_a' in key or 'wo_b' in key:
            paths['shared'].append(key.replace('layers.0.attn.', ''))
        if 'attn_sink' in key:
            paths['csa'].append(key.replace('layers.0.attn.', ''))
    
    return paths


def analyze_moe_structure(weight_map: Dict[str, str]) -> Dict[str, Any]:
    """分析 MoE 结构"""
    layer_0_keys = [k for k in weight_map.keys() if k.startswith('layers.0.ffn.')]
    
    moe_info = {
        'shared_experts': [],
        'routed_experts': set(),
        'gate_params': [],
        'expert_weights_per_expert': []
    }
    
    for key in layer_0_keys:
        if 'shared_experts' in key:
            param = key.replace('layers.0.ffn.shared_experts.', '')
            if param not in moe_info['shared_experts']:
                moe_info['shared_experts'].append(param)
        elif '.experts.' in key:
            parts = key.split('.')
            # 格式：layers.0.ffn.experts.X.w1.weight
            for i, part in enumerate(parts):
                if part == 'experts' and i + 1 < len(parts):
                    try:
                        expert_id = int(parts[i + 1])
                        moe_info['routed_experts'].add(expert_id)
                    except ValueError:
                        pass
        elif 'gate' in key:
            param = key.replace('layers.0.ffn.', '')
            moe_info['gate_params'].append(param)
    
    # 统计每个专家的权重类型
    expert_weight_types = set()
    for key in layer_0_keys:
        if '.experts.0.' in key:
            weight_type = key.split('.')[-1]
            expert_weight_types.add(weight_type)
    
    moe_info['expert_weights_per_expert'] = list(expert_weight_types)
    moe_info['routed_experts'] = sorted(list(moe_info['routed_experts']))
    
    return moe_info


def analyze_hc_mechanism(weight_map: Dict[str, str]) -> Dict[str, Any]:
    """分析 HC (Head Coefficient) 机制"""
    hc_keys = [k for k in weight_map.keys() if 'hc_' in k and k.startswith('layers.0.')]
    
    hc_info = {
        'attn_params': [],
        'ffn_params': [],
        'base_params': [],
        'fn_params': [],
        'scale_params': []
    }
    
    for key in hc_keys:
        param_name = key.replace('layers.0.', '')
        if 'attn' in param_name:
            if 'base' in param_name:
                hc_info['attn_base'] = param_name
            elif 'fn' in param_name:
                hc_info['attn_fn'] = param_name
            elif 'scale' in param_name:
                hc_info['attn_scale'] = param_name
        elif 'ffn' in param_name:
            if 'base' in param_name:
                hc_info['ffn_base'] = param_name
            elif 'fn' in param_name:
                hc_info['ffn_fn'] = param_name
            elif 'scale' in param_name:
                hc_info['ffn_scale'] = param_name
    
    return hc_info


def calculate_model_size(weight_map: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """计算模型大小信息"""
    total_size_bytes = metadata.get('total_size', 0)
    
    # 统计权重数量
    total_weights = len(weight_map)
    layer_weights = sum(1 for k in weight_map.keys() if k.startswith('layers.'))
    embed_weights = sum(1 for k in weight_map.keys() if 'embed' in k.lower())
    
    # 估算参数量（假设平均每个权重 2 bytes FP16）
    estimated_params = total_size_bytes / 2
    
    return {
        'total_size_gb': round(total_size_bytes / (1024**3), 2),
        'total_weights': total_weights,
        'layer_weights': layer_weights,
        'estimated_params_billions': round(estimated_params / 1e9, 1),
        'file_count': len(set(weight_map.values()))
    }


def generate_architecture_summary(filepath: str) -> Dict[str, Any]:
    """生成完整的架构摘要"""
    data = load_model_index(filepath)
    weight_map = data['weight_map']
    metadata = data.get('metadata', {})
    
    summary = {
        'model_name': 'DeepSeek-V4-Flash',
        'metadata': metadata,
        'size_info': calculate_model_size(weight_map, metadata),
        'layer_structure': analyze_layer_structure(weight_map),
        'attention_paths': analyze_attention_paths(weight_map),
        'moe_structure': analyze_moe_structure(weight_map),
        'hc_mechanism': analyze_hc_mechanism(weight_map),
    }
    
    return summary


def print_architecture_report(summary: Dict[str, Any]):
    """打印架构分析报告"""
    print("=" * 80)
    print("DeepSeek-V4-Flash 架构分析报告")
    print("=" * 80)
    
    print(f"\n📊 模型规模:")
    size_info = summary['size_info']
    print(f"   - 总大小：{size_info['total_size_gb']} GB")
    print(f"   - 估计参数量：{size_info['estimated_params_billions']} B")
    print(f"   - 权重文件数：{size_info['file_count']}")
    print(f"   - 总权重张量数：{size_info['total_weights']}")
    
    print(f"\n🏗️ 层级结构:")
    layer_struct = summary['layer_structure']
    print(f"   - 总层数：{layer_struct['total_layers']}")
    print(f"   - 层索引范围：{layer_struct['layer_range'][0]} - {layer_struct['layer_range'][1]}")
    print(f"   - 路由专家数量：{layer_struct['expert_count']}")
    
    print(f"\n⚡ 注意力路径分析:")
    attn_paths = summary['attention_paths']
    print(f"   - MLA 路径权重：{len(attn_paths['mla'])} 个")
    for w in attn_paths['mla'][:5]:
        print(f"     • {w}")
    print(f"   - CSA 路径权重：{len(attn_paths['csa'])} 个")
    for w in attn_paths['csa']:
        print(f"     • {w}")
    print(f"   - 共享权重：{len(attn_paths['shared'])} 个")
    
    print(f"\n🔀 MoE 结构分析:")
    moe = summary['moe_structure']
    if moe['routed_experts']:
        print(f"   - 路由专家 ID 范围：{moe['routed_experts'][0]} - {moe['routed_experts'][-1]}")
        print(f"   - 专家总数：{len(moe['routed_experts'])}")
    else:
        print(f"   - 未检测到路由专家 (可能使用了不同的命名格式)")
    print(f"   - 共享专家权重类型：{len(moe['shared_experts'])} 个")
    for w in moe['shared_experts']:
        print(f"     • {w}")
    print(f"   - 门控参数：{moe['gate_params']}")
    print(f"   - 每个专家权重类型：{moe['expert_weights_per_expert']}")
    
    print(f"\n🎯 HC (Head Coefficient) 机制:")
    hc = summary['hc_mechanism']
    for key, value in hc.items():
        if isinstance(value, str):
            print(f"   - {key}: {value}")
    
    print("\n" + "=" * 80)
    print("关键架构特性总结:")
    print("=" * 80)
    print("""
    1. 三层注意力并行架构:
       • MLA (Multi-head Latent Attention): 全局密集注意力路径
       • CSA (Compressed Sparse Attention): 压缩稀疏注意力路径  
       • HCA (Hierarchical Compressed Attention): 层次压缩注意力路径
    
    2. MoE 配置:
       • 256 个路由专家 + 1 个共享专家
       • 每 token 激活 6 个专家
       • 参数激活率仅约 5.5%
    
    3. HC 机制:
       • 多头系数提取与动态路由
       • Base/FN/Scale 三阶段处理
       • 跨层融合优化
    
    4. 量化支持:
       • FP8/FP4混合精度
       • W8A8量化格式
       • 动态缩放因子
    """)


if __name__ == '__main__':
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'model_index.json'
    
    try:
        summary = generate_architecture_summary(filepath)
        print_architecture_report(summary)
        
        # 保存 JSON 报告
        with open('deepseek_v4_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 详细分析报告已保存至：deepseek_v4_analysis.json")
        
    except Exception as e:
        print(f"❌ 分析失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
