"""
LADO 知识库核心模块
实现 KnowledgeBase 和 KnowledgeUnit 类，支持三维增强知识检索
"""
import json
import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class KnowledgeUnit:
    """
    知识单元：表示单个参数的完整知识
    支持 LADO 三维增强：症状映射、参数依赖、硬约束规则
    """
    # 基础字段
    name: str
    category: str
    context: str  # user/sighup/postmaster
    unit: Optional[str] = None
    description: str = ""
    
    # 三维增强字段
    symptoms: List[str] = field(default_factory=list)  # 症状列表
    dependencies: List[str] = field(default_factory=list)  # 依赖参数
    constraints: Dict = field(default_factory=dict)  # 约束规则 {hard_rule, min, max}
    
    # 调优相关
    tuning_tips: str = ""
    default_formula: Optional[str] = None
    related_knobs: List[str] = field(default_factory=list)  # 相关参数（资源竞争）
    
    # 元数据
    source: str = "structured_knowledge"  # 知识来源
    
    @classmethod
    def from_json(cls, filepath: str) -> 'KnowledgeUnit':
        """从 JSON 文件加载知识单元"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            name=data.get('name', ''),
            category=data.get('category', 'General'),
            context=data.get('context', 'user'),
            unit=data.get('unit'),
            description=data.get('description', ''),
            symptoms=data.get('symptoms', []),
            dependencies=data.get('dependencies', []),
            constraints=data.get('constraints', {}),
            tuning_tips=data.get('tuning_tips', ''),
            default_formula=data.get('default_formula'),
            related_knobs=data.get('related_knobs', []),
            source='structured_knowledge'
        )
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'name': self.name,
            'category': self.category,
            'context': self.context,
            'unit': self.unit,
            'description': self.description,
            'symptoms': self.symptoms,
            'dependencies': self.dependencies,
            'constraints': self.constraints,
            'tuning_tips': self.tuning_tips,
            'default_formula': self.default_formula,
            'related_knobs': self.related_knobs,
            'source': self.source
        }
    
    def get_embedding_texts(self) -> Dict[str, str]:
        """
        生成多视角嵌入文本
        用于支持不同类型的检索（定义检索、症状检索、调优检索）
        """
        texts = {}
        
        # 1. 定义视角（Definition View）
        definition_parts = [
            f"{self.name}",
            f"类别: {self.category}",
            f"描述: {self.description}"
        ]
        if self.unit:
            definition_parts.append(f"单位: {self.unit}")
        texts['definition'] = " | ".join(definition_parts)
        
        # 2. 症状视角（Symptom View）
        if self.symptoms:
            symptom_parts = [f"{self.name} 可解决的性能问题:"]
            symptom_parts.extend(self.symptoms)
            texts['symptom'] = " ".join(symptom_parts)
        
        # 3. 调优视角（Tuning View）
        tuning_parts = [f"{self.name} 调优建议:"]
        if self.tuning_tips:
            tuning_parts.append(self.tuning_tips)
        if self.default_formula:
            tuning_parts.append(f"默认公式: {self.default_formula}")
        texts['tuning'] = " ".join(tuning_parts)
        
        # 4. 依赖视角（Dependency View）
        if self.dependencies or self.related_knobs:
            dep_parts = [f"{self.name} 相关参数:"]
            if self.dependencies:
                dep_parts.append(f"依赖: {', '.join(self.dependencies)}")
            if self.related_knobs:
                dep_parts.append(f"相关: {', '.join(self.related_knobs)}")
            texts['dependency'] = " ".join(dep_parts)
        
        return texts
    
    def matches_symptom(self, symptom_query: str) -> bool:
        """
        检查是否匹配特定症状
        用于症状驱动的参数检索
        """
        symptom_query_lower = symptom_query.lower()
        for symptom in self.symptoms:
            if symptom_query_lower in symptom.lower() or symptom.lower() in symptom_query_lower:
                return True
        return False
    
    def get_constraint_rule(self) -> Optional[str]:
        """获取硬约束规则"""
        return self.constraints.get('hard_rule')
    
    def needs_restart(self) -> bool:
        """判断修改后是否需要重启数据库"""
        return self.context == 'postmaster'


class KnowledgeBase:
    """
    LADO 知识库管理类
    负责加载、索引、检索结构化知识单元
    """
    
    def __init__(self, db: str = 'postgres', base_path: str = './knowledge_collection'):
        self.db = db
        self.base_path = base_path
        self.knowledge_units: Dict[str, KnowledgeUnit] = {}  # name -> KnowledgeUnit
        
        # 索引结构
        self.symptom_index: Dict[str, Set[str]] = {}  # symptom_keyword -> set of knob names
        self.category_index: Dict[str, Set[str]] = {}  # category -> set of knob names
        self.dependency_graph: Dict[str, Set[str]] = {}  # knob -> set of dependent knobs
        
        # 加载知识库
        self._load_knowledge_units()
        self._build_indices()
    
    def _load_knowledge_units(self):
        """加载所有结构化知识单元"""
        struct_path = os.path.join(self.base_path, self.db, 'structured_knowledge')
        
        if not os.path.exists(struct_path):
            print(f"警告：未找到结构化知识目录 {struct_path}")
            return
        
        loaded_count = 0
        for subdir in ['normal', 'special', 'max']:
            subdir_path = os.path.join(struct_path, subdir)
            if not os.path.exists(subdir_path):
                continue
            
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(subdir_path, filename)
                    try:
                        unit = KnowledgeUnit.from_json(filepath)
                        self.knowledge_units[unit.name] = unit
                        loaded_count += 1
                    except Exception as e:
                        print(f"警告：加载 {filename} 失败: {e}")
        
        print(f"✓ 加载了 {loaded_count} 个知识单元")
    
    def _build_indices(self):
        """构建多维索引"""
        print("构建知识库索引...")
        
        for name, unit in self.knowledge_units.items():
            # 1. 症状索引（关键词 -> 参数）
            for symptom in unit.symptoms:
                # 提取关键词（简单分词）
                keywords = self._extract_keywords(symptom)
                for keyword in keywords:
                    if keyword not in self.symptom_index:
                        self.symptom_index[keyword] = set()
                    self.symptom_index[keyword].add(name)
            
            # 2. 分类索引
            if unit.category not in self.category_index:
                self.category_index[unit.category] = set()
            self.category_index[unit.category].add(name)
            
            # 3. 依赖图
            for dep in unit.dependencies:
                if dep not in self.dependency_graph:
                    self.dependency_graph[dep] = set()
                self.dependency_graph[dep].add(name)
        
        print(f"✓ 症状索引: {len(self.symptom_index)} 个关键词")
        print(f"✓ 分类索引: {len(self.category_index)} 个类别")
        print(f"✓ 依赖图: {len(self.dependency_graph)} 个节点")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词
        简单实现：转小写 + 分割
        """
        # 移除标点符号
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # 分词并过滤停用词
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'}
        keywords = [w for w in text.split() if w and w not in stopwords and len(w) > 2]
        return keywords
    
    def get_unit(self, knob_name: str) -> Optional[KnowledgeUnit]:
        """获取指定参数的知识单元"""
        return self.knowledge_units.get(knob_name)
    
    def search_by_symptom(self, symptom_query: str, top_k: int = 10) -> List[KnowledgeUnit]:
        """
        根据症状搜索相关参数（症状 -> 参数反向映射）
        
        Args:
            symptom_query: 症状描述（如 "high IO wait", "OOM error"）
            top_k: 返回前 k 个最相关的参数
            
        Returns:
            匹配的知识单元列表
        """
        keywords = self._extract_keywords(symptom_query)
        
        # 统计每个参数的匹配分数
        scores = {}
        for keyword in keywords:
            if keyword in self.symptom_index:
                for knob_name in self.symptom_index[keyword]:
                    scores[knob_name] = scores.get(knob_name, 0) + 1
        
        # 按分数排序
        ranked_knobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [self.knowledge_units[knob_name] for knob_name, _ in ranked_knobs 
                if knob_name in self.knowledge_units]
    
    def get_by_category(self, category: str) -> List[KnowledgeUnit]:
        """获取指定类别的所有参数"""
        if category not in self.category_index:
            return []
        return [self.knowledge_units[name] for name in self.category_index[category]]
    
    def get_dependent_knobs(self, knob_name: str) -> List[KnowledgeUnit]:
        """
        获取依赖于指定参数的所有参数
        
        Args:
            knob_name: 参数名
            
        Returns:
            依赖该参数的知识单元列表
        """
        if knob_name not in self.dependency_graph:
            return []
        
        return [self.knowledge_units[name] for name in self.dependency_graph[knob_name]
                if name in self.knowledge_units]
    
    def get_dependency_chain(self, knob_name: str) -> List[str]:
        """
        获取参数的完整依赖链
        
        Args:
            knob_name: 参数名
            
        Returns:
            依赖链（从当前参数到所有依赖的参数）
        """
        if knob_name not in self.knowledge_units:
            return []
        
        unit = self.knowledge_units[knob_name]
        chain = []
        visited = set()
        
        def _dfs(name):
            if name in visited:
                return
            visited.add(name)
            chain.append(name)
            
            if name in self.knowledge_units:
                for dep in self.knowledge_units[name].dependencies:
                    _dfs(dep)
        
        _dfs(knob_name)
        return chain
    
    def get_all_units(self) -> List[KnowledgeUnit]:
        """获取所有知识单元"""
        return list(self.knowledge_units.values())
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        stats = {
            'total_units': len(self.knowledge_units),
            'categories': {},
            'avg_symptoms_per_unit': 0,
            'units_with_dependencies': 0,
            'units_with_constraints': 0,
            'units_needing_restart': 0
        }
        
        total_symptoms = 0
        for unit in self.knowledge_units.values():
            # 分类统计
            stats['categories'][unit.category] = stats['categories'].get(unit.category, 0) + 1
            
            # 症状统计
            total_symptoms += len(unit.symptoms)
            
            # 依赖统计
            if unit.dependencies:
                stats['units_with_dependencies'] += 1
            
            # 约束统计
            if unit.constraints:
                stats['units_with_constraints'] += 1
            
            # 重启需求统计
            if unit.needs_restart():
                stats['units_needing_restart'] += 1
        
        stats['avg_symptoms_per_unit'] = total_symptoms / len(self.knowledge_units) if self.knowledge_units else 0
        
        return stats


if __name__ == '__main__':
    # 测试知识库
    kb = KnowledgeBase(db='postgres')
    
    print("\n=== 知识库统计 ===")
    stats = kb.get_statistics()
    print(f"总参数数: {stats['total_units']}")
    print(f"平均症状数: {stats['avg_symptoms_per_unit']:.2f}")
    print(f"有依赖的参数: {stats['units_with_dependencies']}")
    print(f"有约束的参数: {stats['units_with_constraints']}")
    print(f"需要重启的参数: {stats['units_needing_restart']}")
    print(f"\n分类分布:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print("\n=== 症状检索测试 ===")
    results = kb.search_by_symptom("high IO wait", top_k=5)
    print(f"查询: 'high IO wait'")
    print(f"找到 {len(results)} 个相关参数:")
    for unit in results:
        print(f"  - {unit.name} ({unit.category})")
        print(f"    症状: {unit.symptoms[:2]}")
    
    print("\n=== 依赖链测试 ===")
    test_knob = 'work_mem'
    if test_knob in kb.knowledge_units:
        chain = kb.get_dependency_chain(test_knob)
        print(f"{test_knob} 的依赖链: {' -> '.join(chain)}")
