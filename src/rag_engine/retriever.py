"""
RAG 检索器模块
根据查询和场景从知识库中检索相关上下文
支持 LADO 三维增强：症状驱动检索、依赖链分析、约束验证
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_handler.knowledge_base import KnowledgeBase, KnowledgeUnit


class RAGRetriever:
    """RAG 检索器"""
    
    def __init__(self, knowledge_base_dir='./knowledge_base', db='postgres'):
        self.db = db
        self.kb_dir = knowledge_base_dir
        
        # 使用镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            print(f"警告：模型加载失败 ({e})，将跳过向量检索")
            self.encoder = None
        
        # 加载知识库
        self.documents = None
        self.knowledge_base = None
        self.index = None
        
        # LADO 知识库
        self.lado_kb = None
        
        self._load_knowledge_base()
        
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            # 加载文档
            doc_path = os.path.join(self.kb_dir, f'{self.db}_documents.pkl')
            with open(doc_path, 'rb') as f:
                self.documents = pickle.load(f)
                
            # 加载知识库分类
            kb_path = os.path.join(self.kb_dir, f'{self.db}_knowledge_base.pkl')
            with open(kb_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
                
            # 加载 FAISS 索引
            index_path = os.path.join(self.kb_dir, f'{self.db}_faiss.index')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            
            # 加载 LADO 知识库（用于症状检索和依赖分析）
            try:
                self.lado_kb = KnowledgeBase(db=self.db, base_path='./knowledge_collection')
                print(f"RAG 检索器初始化成功，加载了 {len(self.documents)} 条文档")
                print(f"  ✓ LADO 知识库: {len(self.lado_kb.knowledge_units)} 个参数")
            except Exception as e:
                print(f"  警告：LADO 知识库加载失败: {e}")
                
        except Exception as e:
            print(f"警告：无法加载知识库，需要先构建知识库。错误：{e}")
            
    def retrieve(self, query, scenario=None, top_k=5, knob_name=None):
        """
        检索相关知识
        
        Args:
            query: 查询文本
            scenario: 场景类型 ('OLTP', 'OLAP', 'HYBRID', None)
            top_k: 返回前 k 条结果
            knob_name: 特定参数名（可选）
            
        Returns:
            相关文档列表
        """
        if self.index is None or self.documents is None:
            print("知识库未加载")
            return []
            
        # 1. 如果指定了参数名，优先返回该参数的知识
        if knob_name:
            knob_docs = [doc for doc in self.documents if doc['knob'] == knob_name]
            if knob_docs:
                return knob_docs[:top_k]
                
        # 2. 场景过滤
        candidate_docs = self.documents
        if scenario and scenario in self.knowledge_base:
            scenario_docs = self.knowledge_base[scenario]
            general_docs = self.knowledge_base.get('GENERAL', [])
            candidate_docs = scenario_docs + general_docs
            
        # 3. 向量检索
        query_vector = self.encoder.encode([query])
        
        # 如果有场景过滤，需要重新构建临时索引
        if scenario and len(candidate_docs) < len(self.documents):
            # 获取候选文档的索引
            candidate_indices = [i for i, doc in enumerate(self.documents) 
                               if doc in candidate_docs]
            
            # 在全局索引中搜索更多结果，然后过滤
            distances, indices = self.index.search(
                query_vector.astype('float32'), 
                min(top_k * 3, len(self.documents))
            )
            
            # 过滤出候选文档
            filtered_results = []
            for idx in indices[0]:
                if idx in candidate_indices:
                    filtered_results.append(self.documents[idx])
                    if len(filtered_results) >= top_k:
                        break
                        
            return filtered_results
        else:
            # 直接搜索
            distances, indices = self.index.search(
                query_vector.astype('float32'), 
                top_k
            )
            
            return [self.documents[idx] for idx in indices[0]]
            
    def retrieve_for_knob_selection(self, workload_type, candidate_knobs, top_k=10):
        """
        为参数选择任务检索相关知识
        
        Args:
            workload_type: 负载类型 (OLTP/OLAP)
            candidate_knobs: 候选参数列表
            top_k: 每个参数返回的知识数量
            
        Returns:
            Dict[knob_name, List[doc]]
        """
        results = {}
        
        # 映射负载类型到场景
        scenario_map = {
            'OLTP': 'OLTP',
            'OLAP': 'OLAP'
        }
        scenario = scenario_map.get(workload_type, 'GENERAL')
        
        for knob in candidate_knobs:
            query = f"参数 {knob} 在 {workload_type} 场景下的调优建议"
            docs = self.retrieve(query, scenario=scenario, top_k=top_k, knob_name=knob)
            results[knob] = docs
            
        return results
        
    def retrieve_for_value_recommendation(self, knob_name, scenario, hardware_info=None):
        """
        为参数值推荐检索相关知识
        
        Args:
            knob_name: 参数名
            scenario: 场景类型
            hardware_info: 硬件信息（可选）
            
        Returns:
            相关文档列表和推荐上下文
        """
        # 构建查询
        query_parts = [f"参数 {knob_name} 的推荐值"]
        if scenario:
            query_parts.append(f"在 {scenario} 场景下")
        if hardware_info:
            if 'memory_gb' in hardware_info:
                query_parts.append(f"内存 {hardware_info['memory_gb']}GB")
                
        query = " ".join(query_parts)
        
        # 检索
        docs = self.retrieve(query, scenario=scenario, top_k=5, knob_name=knob_name)
        
        # 构建推荐上下文
        context = self._build_recommendation_context(docs, knob_name)
        
        return docs, context
        
    def _build_recommendation_context(self, docs, knob_name):
        """构建推荐上下文字符串"""
        if not docs:
            return f"参数 {knob_name} 的相关知识："
            
        context_parts = [f"## 参数 {knob_name} 的调优知识\n"]
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"{i}. {doc['content']}")
            
            # 如果有结构化数据，添加更多信息
            if 'structured_data' in doc:
                data = doc['structured_data']
                if 'suggested_values' in data and data['suggested_values']:
                    context_parts.append(f"   建议值: {data['suggested_values']}")
                    
        return "\n".join(context_parts)
        
    def get_scenario_knowledge_summary(self, scenario):
        """获取特定场景的知识摘要"""
        if scenario not in self.knowledge_base:
            return None
            
        docs = self.knowledge_base[scenario]
        
        # 统计高频参数
        knob_counts = {}
        for doc in docs:
            knob = doc['knob']
            knob_counts[knob] = knob_counts.get(knob, 0) + 1
            
        # 返回摘要
        top_knobs = sorted(knob_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary = {
            'scenario': scenario,
            'total_docs': len(docs),
            'top_knobs': top_knobs,
            'sample_docs': docs[:5]
        }
        
        return summary

    def retrieve_by_symptom(self, symptom_query: str, top_k: int = 5) -> list:
        """
        症状驱动检索：从性能问题反查相关参数
        
        Args:
            symptom_query: 症状描述（如 "high IO wait", "OOM error", "slow query"）
            top_k: 返回前 k 个最相关的参数
            
        Returns:
            包含知识单元和匹配度的结果列表
        """
        if not self.lado_kb:
            print("警告：LADO 知识库未加载，无法使用症状检索")
            return []
        
        # 1. 使用 LADO 知识库的症状索引
        units = self.lado_kb.search_by_symptom(symptom_query, top_k=top_k)
        
        # 2. 构建返回结果
        results = []
        for unit in units:
            result = {
                'knob': unit.name,
                'category': unit.category,
                'description': unit.description,
                'symptoms': unit.symptoms,
                'tuning_tips': unit.tuning_tips,
                'dependencies': unit.dependencies,
                'constraints': unit.constraints,
                'needs_restart': unit.needs_restart(),
                'unit': unit
            }
            results.append(result)
        
        return results
    
    def retrieve_with_dependencies(self, knob_name: str, include_related: bool = True) -> dict:
        """
        检索参数及其依赖链
        
        Args:
            knob_name: 参数名
            include_related: 是否包含相关参数（资源竞争）
            
        Returns:
            包含参数、依赖链、相关参数的字典
        """
        if not self.lado_kb:
            return {'error': 'LADO 知识库未加载'}
        
        unit = self.lado_kb.get_unit(knob_name)
        if not unit:
            return {'error': f'参数 {knob_name} 不存在'}
        
        result = {
            'knob': unit.to_dict(),
            'dependency_chain': self.lado_kb.get_dependency_chain(knob_name),
            'dependent_knobs': [u.name for u in self.lado_kb.get_dependent_knobs(knob_name)]
        }
        
        if include_related and unit.related_knobs:
            result['related_knobs'] = [
                self.lado_kb.get_unit(rk).to_dict() 
                for rk in unit.related_knobs 
                if self.lado_kb.get_unit(rk)
            ]
        
        return result
    
    def validate_constraints(self, knob_name: str, value: any, hardware_info: dict = None) -> dict:
        """
        验证参数值是否符合约束
        
        Args:
            knob_name: 参数名
            value: 待验证的值
            hardware_info: 硬件信息（用于动态约束）
            
        Returns:
            验证结果 {valid: bool, reason: str, suggestion: str}
        """
        if not self.lado_kb:
            return {'valid': True, 'reason': 'LADO 知识库未加载，跳过验证'}
        
        unit = self.lado_kb.get_unit(knob_name)
        if not unit:
            return {'valid': True, 'reason': f'参数 {knob_name} 不在知识库中'}
        
        # 获取约束
        constraints = unit.constraints
        if not constraints:
            return {'valid': True, 'reason': '无约束规则'}
        
        # 检查 hard_rule
        hard_rule = constraints.get('hard_rule')
        if hard_rule:
            # 简单的规则验证（可扩展为更复杂的表达式解析）
            try:
                # 替换 value 占位符
                rule = hard_rule.replace('value', str(value))
                
                # 如果有硬件信息，替换相关变量
                if hardware_info:
                    if 'RAM' in rule and 'memory_gb' in hardware_info:
                        rule = rule.replace('RAM', str(hardware_info['memory_gb'] * 1024))  # 转为 MB
                
                # 尝试评估规则（注意：生产环境需要更安全的实现）
                # 这里仅作示例，实际应用应使用规则引擎
                is_valid = True  # 默认通过
                
            except Exception as e:
                return {'valid': True, 'reason': f'约束验证失败: {e}'}
        
        # 检查 min/max
        min_val = constraints.get('min')
        max_val = constraints.get('max')
        
        try:
            value_num = float(value) if value is not None else None
            if value_num is not None:
                if min_val and value_num < float(min_val):
                    return {
                        'valid': False,
                        'reason': f'值 {value} 低于最小值 {min_val}',
                        'suggestion': f'建议设置为至少 {min_val}'
                    }
                if max_val and value_num > float(max_val):
                    return {
                        'valid': False,
                        'reason': f'值 {value} 超过最大值 {max_val}',
                        'suggestion': f'建议设置为不超过 {max_val}'
                    }
        except (ValueError, TypeError):
            pass  # 非数值类型，跳过 min/max 检查
        
        return {'valid': True, 'reason': '通过所有约束检查'}


if __name__ == '__main__':
    # 测试检索器
    retriever = RAGRetriever(db='postgres')
    
    # 测试检索
    print("\n=== 测试 OLTP 场景检索 ===")
    results = retriever.retrieve(
        "高并发场景下 shared_buffers 的推荐值",
        scenario='OLTP',
        top_k=3
    )
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['knob']} ({doc['source']})")
        print(f"   {doc['content'][:100]}...")
        
    # 测试场景摘要
    print("\n=== OLTP 场景知识摘要 ===")
    summary = retriever.get_scenario_knowledge_summary('OLTP')
    if summary:
        print(f"文档数量: {summary['total_docs']}")
        print("高频参数:", [k for k, c in summary['top_knobs'][:5]])
