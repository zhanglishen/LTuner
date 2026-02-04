from openai import OpenAI, APIError
import re
import json
import tiktoken
import os
import sys

# 添加 RAG 支持
try:
    from rag_engine.retriever import RAGRetriever
    from scenario_engine.classifier import ScenarioClassifier
    from scenario_engine.prompt_templates import PromptTemplates
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("警告：RAG模块未找到，将使用原始GPT模式")

class GPT:
    def __init__(self, api_base, api_key, model="gpt-4o-mini", use_rag=False, db="postgres"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.money = 0
        self.token = 0
        self.cur_token = 0
        self.cur_money = 0
        
        # RAG 相关
        self.use_rag = use_rag and RAG_AVAILABLE
        self.db = db
        self.rag_retriever = None
        self.scenario_classifier = None
        
        if self.use_rag:
            self._init_rag()
            
    def _init_rag(self):
        """初始化 RAG 组件"""
        try:
            # 检查知识库是否存在
            kb_dir = './knowledge_base'
            if os.path.exists(os.path.join(kb_dir, f'{self.db}_documents.pkl')):
                # 使用镜像源
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.rag_retriever = RAGRetriever(knowledge_base_dir=kb_dir, db=self.db)
                self.scenario_classifier = ScenarioClassifier()
                print(f"RAG 增强模式已启用 (数据库: {self.db})")
            else:
                print(f"警告：未找到知识库，请先运行 knowledge_builder.py 构建知识库")
                self.use_rag = False
        except Exception as e:
            print(f"RAG 初始化失败: {e}")
            self.use_rag = False

    def get_GPT_response_json(self, prompt, json_format=True, scenario=None, knob_name=None): # This function returns the GPT response, which can be specified to return json or string format
        # RAG 增强：如果启用RAG且提供了场景信息，则增强prompt
        if self.use_rag and self.rag_retriever and (scenario or knob_name):
            prompt = self._enhance_prompt_with_rag(prompt, scenario, knob_name)
            
        client = OpenAI(api_key=self.api_key, base_url = self.api_base)
        if json_format: # json
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You should output JSON."},
                    {'role':'user', 'content':prompt}],
                model=self.model, 
                response_format={"type": "json_object"}, 
                temperature=0.5,
            )
            # print(response)
            ans = response.choices[0].message.content
            completion = json.loads(ans)  # Convert to json object
            
        else: # string
            response = client.chat.completions.create(
                messages=[
                    {'role':'user', 'content':prompt}],
                model=self.model, 
                temperature=1,     
            )
            completion = response.choices[0].message.content
        return completion
    
    def calc_token(self, in_text, out_text=""):
        if isinstance(out_text, dict):
            out_text = json.dumps(out_text)
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(out_text+in_text))

    def calc_money(self, in_text, out_text):
        """money for gpt4"""
        if self.model == "gpt-4":
            return (self.calc_token(in_text) * 0.03 + self.calc_token(out_text) * 0.06) / 1000
        elif self.model == "gpt-3.5-turbo":
            return (self.calc_token(in_text) * 0.0015 + self.calc_token(out_text) * 0.002) / 1000
        elif self.model == "gpt-4-1106-preview" or self.model == "gpt-4-1106-vision-preview":
            return (self.calc_token(in_text) * 0.01 + self.calc_token(out_text) * 0.03) / 1000
        else:
            return 0 

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _enhance_prompt_with_rag(self, prompt, scenario=None, knob_name=None):
        """使用 RAG 增强 prompt"""
        try:
            # 从 prompt 中提取查询意图
            query = self._extract_query_from_prompt(prompt)
            
            # 检索相关知识
            docs = self.rag_retriever.retrieve(
                query=query,
                scenario=scenario,
                top_k=3,
                knob_name=knob_name
            )
            
            if docs:
                # 构建增强上下文
                context = "\n\n## 相关知识库参考\n"
                for i, doc in enumerate(docs, 1):
                    context += f"{i}. {doc['content'][:200]}...\n"
                    
                # 将上下文插入 prompt
                enhanced_prompt = prompt + context
                print(f"[RAG] 已为 prompt 添加 {len(docs)} 条知识")
                return enhanced_prompt
        except Exception as e:
            print(f"[RAG] 增强失败: {e}")
            
        return prompt
        
    def _extract_query_from_prompt(self, prompt):
        """从 prompt 中提取关键查询信息"""
        # 简单提取：取 prompt 的前200个字符作为查询
        return prompt[:200]
        
    def enable_rag(self, enable=True):
        """启用或禁用 RAG"""
        if RAG_AVAILABLE:
            self.use_rag = enable
            if enable and not self.rag_retriever:
                self._init_rag()
        else:
            print("警告：RAG 模块不可用")
            
    def classify_scenario(self, workload_stats=None, benchmark=None):
        """分类场景"""
        if not self.scenario_classifier:
            self.scenario_classifier = ScenarioClassifier()
            
        if benchmark:
            return self.scenario_classifier.classify_by_benchmark(benchmark)
        elif workload_stats:
            return self.scenario_classifier.classify(workload_stats)
        else:
            return None
    
