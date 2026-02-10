"""
知识库构建模块
从现有 knowledge_collection 提取知识并构建向量索引
支持 LADO 三维增强：症状映射、参数依赖、硬约束规则
"""
import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_handler.knowledge_base import KnowledgeBase, KnowledgeUnit


class KnowledgeBuilder:
    """构建 PostgreSQL 参数调优知识库"""
    
    def __init__(self, db='postgres', model_name='paraphrase-MiniLM-L6-v2'):
        self.db = db
        self.base_path = f"./knowledge_collection/{db}"
        
        # 使用镜像源加速HuggingFace模型下载
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print(f"正在加载Sentence-BERT模型: {model_name}...")
        try:
            self.encoder = SentenceTransformer(model_name)
            print("模型加载成功！")
        except Exception as e:
            print(f"警告：模型加载失败 ({e})，将跳过向量化步骤")
            self.encoder = None
            
        self.knowledge_base = {
            'OLTP': [],
            'OLAP': [],
            'HYBRID': [],
            'GENERAL': []
        }
        self.documents = []
        self.embeddings = None
        self.index = None
        
        # LADO 知识库
        self.lado_kb = None
        self.multi_view_embeddings = {}  # view_name -> embeddings
        
    def build_knowledge_base(self):
        """从现有知识源构建知识库"""
        print("开始构建知识库...")
        
        # 1. 加载 LADO 结构化知识库（优先）
        print("\n[1/4] 加载 LADO 结构化知识库...")
        self._load_lado_knowledge()
        
        # 2. 加载调优湖（tuning_lake）中的知识
        print("\n[2/4] 加载 Tuning Lake 知识...")
        self._load_tuning_lake()
        
        # 3. 加载旧版结构化知识（向后兼容）
        print("\n[3/4] 加载旧版结构化知识...")
        self._load_structured_knowledge()
        
        # 4. 加载候选参数信息
        print("\n[4/4] 加载候选参数列表...")
        self._load_candidate_knobs()
        
        print(f"✓ 知识库构建完成，共 {len(self.documents)} 条文档")
        return self.knowledge_base
    
    def _load_lado_knowledge(self):
        """加载 LADO 结构化知识库"""
        try:
            self.lado_kb = KnowledgeBase(db=self.db, base_path='./knowledge_collection')
            
            # 为每个知识单元创建多个文档（多视角）
            for unit in self.lado_kb.get_all_units():
                # 获取多视角嵌入文本
                embedding_texts = unit.get_embedding_texts()
                
                # 1. 定义视角文档
                if 'definition' in embedding_texts:
                    self.documents.append({
                        'knob': unit.name,
                        'content': embedding_texts['definition'],
                        'source': 'lado_definition',
                        'scenario': self._infer_scenario_from_category(unit.category),
                        'unit': unit,
                        'view': 'definition'
                    })
                
                # 2. 症状视角文档（关键！）
                if 'symptom' in embedding_texts and unit.symptoms:
                    self.documents.append({
                        'knob': unit.name,
                        'content': embedding_texts['symptom'],
                        'source': 'lado_symptom',
                        'scenario': self._infer_scenario_from_category(unit.category),
                        'unit': unit,
                        'view': 'symptom'
                    })
                
                # 3. 调优视角文档
                if 'tuning' in embedding_texts and unit.tuning_tips:
                    self.documents.append({
                        'knob': unit.name,
                        'content': embedding_texts['tuning'],
                        'source': 'lado_tuning',
                        'scenario': self._infer_scenario_from_category(unit.category),
                        'unit': unit,
                        'view': 'tuning'
                    })
                
                # 4. 依赖视角文档
                if 'dependency' in embedding_texts:
                    self.documents.append({
                        'knob': unit.name,
                        'content': embedding_texts['dependency'],
                        'source': 'lado_dependency',
                        'scenario': 'GENERAL',
                        'unit': unit,
                        'view': 'dependency'
                    })
                
                # 添加到场景分类
                scenario = self._infer_scenario_from_category(unit.category)
                for doc in self.documents[-len(embedding_texts):]:
                    self.knowledge_base[scenario].append(doc)
            
            print(f"  ✓ 从 LADO 知识库加载了 {len(self.lado_kb.knowledge_units)} 个参数")
            print(f"  ✓ 生成了 {len(self.documents)} 个多视角文档")
            
        except Exception as e:
            print(f"  警告：加载 LADO 知识库失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _infer_scenario_from_category(self, category: str) -> str:
        """根据类别推断场景"""
        category_lower = category.lower()
        
        # OLTP 相关类别
        if any(kw in category_lower for kw in ['connection', 'lock', 'autovacuum', 'buffer']):
            return 'OLTP'
        
        # OLAP 相关类别
        if any(kw in category_lower for kw in ['planner', 'parallel', 'work', 'maintenance']):
            return 'OLAP'
        
        # WAL/复制相关
        if any(kw in category_lower for kw in ['wal', 'replication', 'archive']):
            return 'HYBRID'
        
        return 'GENERAL'
        
    def _load_tuning_lake(self):
        """加载 tuning_lake 中的调优建议"""
        tuning_lake_path = os.path.join(self.base_path, 'tuning_lake')
        if not os.path.exists(tuning_lake_path):
            print(f"警告：未找到 tuning_lake 目录")
            return
            
        for filename in os.listdir(tuning_lake_path):
            if filename.endswith('.txt'):
                knob_name = filename.replace('.txt', '')
                filepath = os.path.join(tuning_lake_path, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    doc = {
                        'knob': knob_name,
                        'content': content,
                        'source': 'tuning_lake',
                        'scenario': self._infer_scenario(knob_name, content)
                    }
                    self.documents.append(doc)
                    scenario = doc['scenario']
                    self.knowledge_base[scenario].append(doc)
                    
        print(f"从 tuning_lake 加载了 {len(os.listdir(tuning_lake_path))} 个参数的知识")
        
    def _load_structured_knowledge(self):
        """加载结构化知识"""
        struct_path = os.path.join(self.base_path, 'structured_knowledge')
        if not os.path.exists(struct_path):
            return
            
        for subdir in ['normal', 'special']:
            subdir_path = os.path.join(struct_path, subdir)
            if not os.path.exists(subdir_path):
                continue
                
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json'):
                    knob_name = filename.replace('.json', '')
                    filepath = os.path.join(subdir_path, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 构建文档
                    content_parts = []
                    if 'min_value' in data and data['min_value']:
                        content_parts.append(f"最小值: {data['min_value']}")
                    if 'max_value' in data and data['max_value']:
                        content_parts.append(f"最大值: {data['max_value']}")
                    if 'suggested_values' in data and data['suggested_values']:
                        content_parts.append(f"建议值: {data['suggested_values']}")
                        
                    if content_parts:
                        content = f"{knob_name} - " + ", ".join(content_parts)
                        doc = {
                            'knob': knob_name,
                            'content': content,
                            'source': f'structured_{subdir}',
                            'scenario': 'GENERAL',
                            'structured_data': data
                        }
                        self.documents.append(doc)
                        self.knowledge_base['GENERAL'].append(doc)
                        
    def _load_candidate_knobs(self):
        """加载候选参数的基本信息"""
        candidate_path = os.path.join(self.base_path, 'candidate_knobs.txt')
        if os.path.exists(candidate_path):
            # candidate_knobs.txt 是纯文本格式，每行一个参数名
            with open(candidate_path, 'r') as f:
                for line in f:
                    knob_name = line.strip()
                    if knob_name:
                        doc = {
                            'knob': knob_name,
                            'content': f"{knob_name}: PostgreSQL 参数",
                            'source': 'candidate_knobs',
                            'scenario': 'GENERAL'
                        }
                        self.documents.append(doc)
                        self.knowledge_base['GENERAL'].append(doc)
                    
    def _infer_scenario(self, knob_name, content):
        """根据参数名和内容推断适用场景"""
        content_lower = content.lower()
        
        # OLTP 相关关键词
        oltp_keywords = ['connection', 'transaction', 'lock', 'concurrency', 
                         'cache', 'buffer', 'random']
        # OLAP 相关关键词
        olap_keywords = ['sort', 'hash', 'aggregate', 'parallel', 'work_mem',
                        'maintenance', 'sequential', 'batch']
        
        oltp_score = sum(1 for kw in oltp_keywords if kw in content_lower)
        olap_score = sum(1 for kw in olap_keywords if kw in content_lower)
        
        if oltp_score > olap_score:
            return 'OLTP'
        elif olap_score > oltp_score:
            return 'OLAP'
        else:
            return 'GENERAL'
            
    def create_embeddings(self):
        """为所有文档生成向量嵌入"""
        if self.encoder is None:
            print("警告：编码器未初始化，跳过向量生成")
            return None
            
        print("生成向量嵌入...")
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        print(f"生成了 {len(self.embeddings)} 个向量")
        return self.embeddings
        
    def build_faiss_index(self):
        """构建FAISS索引"""
        if self.embeddings is None:
            if self.encoder is None:
                print("跳过FAISS索引构建（编码器未初始化）")
                return None
            self.create_embeddings()
                
        if self.embeddings is None:
            return None
                
        print("构建FAISS索引...")
        dimension = self.embeddings.shape[1]
            
        # 使用L2距离的平面索引（适合小规模数据）
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
            
        print(f"FAISS索引构建完成，维度: {dimension}")
        return self.index
        
    def save(self, save_dir='./knowledge_base'):
        """保存知识库和索引"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文档
        with open(os.path.join(save_dir, f'{self.db}_documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
            
        # 保存知识库
        with open(os.path.join(save_dir, f'{self.db}_knowledge_base.pkl'), 'wb') as f:
            pickle.dump(self.knowledge_base, f)
            
        # 保存嵌入
        if self.embeddings is not None:
            np.save(os.path.join(save_dir, f'{self.db}_embeddings.npy'), self.embeddings)
            
        # 保存 FAISS 索引
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_dir, f'{self.db}_faiss.index'))
            
        print(f"知识库已保存至 {save_dir}")
        
    def load(self, save_dir='./knowledge_base'):
        """加载已保存的知识库和索引"""
        # 加载文档
        with open(os.path.join(save_dir, f'{self.db}_documents.pkl'), 'rb') as f:
            self.documents = pickle.load(f)
            
        # 加载知识库
        with open(os.path.join(save_dir, f'{self.db}_knowledge_base.pkl'), 'rb') as f:
            self.knowledge_base = pickle.load(f)
            
        # 加载嵌入
        embeddings_path = os.path.join(save_dir, f'{self.db}_embeddings.npy')
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            
        # 加载 FAISS 索引
        index_path = os.path.join(save_dir, f'{self.db}_faiss.index')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        print(f"知识库已从 {save_dir} 加载")


if __name__ == '__main__':
    # 测试构建知识库
    builder = KnowledgeBuilder(db='postgres')
    builder.build_knowledge_base()
    builder.create_embeddings()
    builder.build_faiss_index()
    builder.save()
    
    print("\n知识库统计：")
    for scenario, docs in builder.knowledge_base.items():
        print(f"  {scenario}: {len(docs)} 条文档")
