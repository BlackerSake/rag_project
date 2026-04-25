# RAG系统评测方案

## 评估维度

### 0. 兜底能力评估（Fallback Capability）— 无答案场景

| 指标 | 描述 | 评估方法 |
|------|------|----------|
| 拒答准确率 | 当问题超出知识库范围时，系统是否正确拒绝回答 | LLM判断：问题是否在知识库覆盖范围内 |
| 幻觉率 | 系统在无法回答时是否乱编答案 | LLM判断：答案是否包含无法从上下文推断的内容 |
| 安全回复率 | 拒绝回答时是否给出友好的兜底回复 | 规则判断：是否包含"不知道"、"无法回答"等关键词 |

#### 兜底能力评估实现逻辑

```
评测数据准备:
  - 输入: 无答案测试用例 {(unanswerable_question_i, expected_behavior_i)}
  - expected_behavior: "reject" (应该拒绝) 或 "safe_answer" (给出安全回复)

兜底评估流程:

  1. 将问题输入RAG系统，获取回答:
     answer_i = rag_system.answer(question_i)

  2. 拒答准确率评估:
     prompt = f"""判断以下问题是否在知识库的覆盖范围内。
     如果问题涉及知识库中没有的信息（如最新政策、实时数据、个人信息等），应该判定为超出范围。

     问题: {question_i}
     知识库内容片段: {retrieved_context_i}

     输出格式: {{"in_scope": true/false, "reasoning": "..."}}
     """
     in_scope = LLM判断后返回的in_scope

  3. 幻觉率评估（当系统尝试回答时）:
     prompt = f"""判断答案是否包含幻觉内容。
     幻觉定义：答案中包含知识库/上下文无法支持的信息。

     上下文: {retrieved_context_i}
     答案: {answer_i}

     输出格式: {{"has_hallucination": true/false, "hallucinated_parts": [...], "faithfulness_score": 0.0-1.0}}
     """
     hallucination_result = LLM判断后返回

  4. 安全回复率评估:
     safe_keywords = ["不知道", "无法回答", "抱歉", "没有找到", "超出范围", "无法提供"]
     is_safe = any(keyword in answer_i for keyword in safe_keywords)

  5. 汇总兜底能力指标:
     Fallback_Capability = {
       "reject_accuracy": 正确拒答数 / 无答案测试总数,
       "hallucination_rate": 幻觉案例数 / 尝试回答数,
       "safe_reply_rate": 安全回复数 / 拒答总数
     }
```

---

### 1. 检索质量（Retrieval Quality）

| 指标 | 描述 | 计算公式 |
|------|------|----------|
| Recall@K | 相关文档是否出现在Top-K结果中 | `Recall@K = |Rel_docs ∩ TopK_docs| / |Rel_docs|` |
| Precision@K | Top-K中有多少是真正相关的 | `Precision@K = |Rel_docs ∩ TopK_docs| / K` |
| MRR（Mean Reciprocal Rank） | 第一个相关结果的排名倒数均值 | `MRR = (1/R1 + 1/R2 + ... + 1/RN) / N` |
| NDCG | 考虑排序位置的综合指标 | `NDCG@K = DCG@K / IDCG@K` |

#### 检索质量评估实现逻辑

```
评测数据准备:
  - 输入: 查询集 Q = {(query_i, relevant_docs_i)}
  - relevant_docs_i: 第i个查询的相关文档ID集合

检索评估流程:
  1. 对每个查询 query_i 执行知识库检索:
     results_i = KnowledgeBase.search(query_i, k=K)

  2. 提取检索结果中的文档ID集合:
     retrieved_docs_i = {doc.metadata["doc_id"] for doc, score in results_i}

  3. 计算各项指标:
     a) Recall@K:
        - 命中 = len(relevant_docs_i ∩ retrieved_docs_i)
        - Recall@K = 命中 / len(relevant_docs_i)

     b) Precision@K:
        - Precision@K = 命中 / K

     c) MRR:
        - 找到第一个相关文档的排名 position
        - MRR = 1 / position

     d) NDCG@K:
        - DCG@K = Σ( relevance_i / log2(i+1) ), i=1 to K
        - IDCG@K = DCG@K 的最大值（理想排序）
        - NDCG@K = DCG@K / IDCG@K

  4. 汇总所有查询的指标:
     - Avg_Recall@K = Σ Recall@K / |Q|
     - Avg_Precision@K = Σ Precision@K / |Q|
     - Avg_MRR = Σ MRR / |Q|
     - Avg_NDCG@K = Σ NDCG@K / |Q|
```

---

### 2. 生成质量（Generation Quality）

| 指标 | 描述 | 评估方法 |
|------|------|----------|
| 忠实度（Faithfulness） | 答案是否基于检索到的文档，有无幻觉 | LLM判断：答案中每个陈述是否可从上下文中推断 |
| 答案相关性（Answer Relevancy） | 答案是否回答了问题 | LLM判断：答案与问题的语义相关性评分 |
| 上下文相关性（Context Relevancy） | 检索内容是否和问题相关 | LLM判断：检索片段与问题的语义相关性 |
| 上下文召回（Context Recall） | 检索内容是否覆盖了答案所需信息 | LLM判断：参考答案中的信息有多少出现在检索内容中 |

#### 生成质量评估实现逻辑

```
评测数据准备:
  - 输入: 测试用例 {(question_i, ground_truth_answer_i, retrieved_context_i)}
  - retrieved_context_i: 检索阶段返回的上下文文档列表

生成评估流程:

  1. 合并评估（一次LLM调用返回四个分数）:
     prompt = f"""同时评估以下四个维度，返回JSON格式的评分。

     维度1 - 忠实度(Faithfulness): 判断答案中的每个陈述是否可从上下文中推断出来。
     维度2 - 答案相关性(Answer Relevancy): 判断答案是否直接回应了问题，问题是否被完整回答。
     维度3 - 上下文相关性(Context Relevancy): 判断每个检索片段是否与问题相关。
     维度4 - 上下文召回(Context Recall): 检查参考答案中的关键信息点有多少出现在检索上下文中。

     问题: {question_i}
     上下文: {retrieved_context_i}
     答案: {generated_answer_i}
     参考答案: {ground_truth_answer_i}

     输出格式:
     {{
       "faithfulness": 0.0-1.0,
       "answer_relevancy": 0.0-1.0,
       "context_relevancy": 0.0-1.0,
       "context_recall": 0.0-1.0,
       "reasoning": {{
         "faithfulness_detail": "...",
         "answer_relevancy_detail": "...",
         "context_relevancy_detail": "...",
         "context_recall_detail": "..."
       }}
     }}
     """
     result = LLM判断后返回的JSON

  2. 汇总生成质量指标:
     Generation_Quality = {
       "faithfulness": result["faithfulness"],
       "answer_relevancy": result["answer_relevancy"],
       "context_relevancy": result["context_relevancy"],
       "context_recall": result["context_recall"]
     }

#### 生成质量评估实现逻辑（拆分流式，保留兼容）
```

---

### 3. 端到端质量

| 指标 | 描述 | 用途 |
|------|------|------|
| ROUGE | 与参考答案的n-gram重叠度（需中文分词） | 衡量内容覆盖程度 |
| BLEU | 与参考答案的n-gram精确度（需中文分词） | 衡量生成质量 |
| BERTScore | 语义层面的相似度，基于BERT embeddings | 更适合中文场景的语义评估 |

#### 端到端质量评估实现逻辑

```
评测数据准备:
  - 输入: 测试用例 {(question_i, ground_truth_answer_i, generated_answer_i)}

端到端评估流程:

  1. 中文分词预处理:
     import jieba
     ref_tokens = " ".join(jieba.cut(ground_truth_answer_i))  # 用空格连接分词结果
     hyp_tokens = " ".join(jieba.cut(generated_answer_i))

  2. ROUGE-N 计算:
     - ROUGE-N = Σ overlap_count(N-gram in both) / Σ count(N-gram in reference)
     - 常用: ROUGE-1 (unigram), ROUGE-2 (bigram), ROUGE-L (最长公共子序列)

     使用 rouge library:
     from rouge import Rouge
     rouge = Rouge()
     scores = rouge.get_scores(hyp_tokens, ref_tokens)  # 分词后计算
     rouge_1 = scores[0]['rouge-1']['f']
     rouge_2 = scores[0]['rouge-2']['f']
     rouge_l = scores[0]['rouge-l']['f']

  3. BLEU 计算:
     - BLEU = BP * exp(Σ w_n * log p_n)
     - BP: brevity penalty (惩罚过短翻译)
     - p_n: n-gram精确度

     使用 nltk.translate.bleu_score:
     from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
     reference = [ref_tokens.split()]  # 二维列表，已分词
     hypothesis = hyp_tokens.split()
     smoothie = SmoothingFunction().method1
     bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

  4. BERTScore 计算（更适合中文）:
     from bert_score import score as bert_score

     # 中英文都适用，基于语义embedding计算相似度
     P, R, F1 = bert_score(
       [generated_answer_i],
       [ground_truth_answer_i],
       lang="zh",  # 或 "en"
       rescale_with_baseline=True
     )
     bertscore_f1 = F1.item()

  5. 汇总端到端指标:
     E2E_Quality = {
       "rouge_1": rouge_1,
       "rouge_2": rouge_2,
       "rouge_l": rouge_l,
       "bleu": bleu_score,
       "bertscore_f1": bertscore_f1
     }
```

---

## 评测系统架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Evaluation Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Retrieval    │    │ Generation    │    │   E2E        │          │
│  │ Evaluator    │    │ Evaluator    │    │   Evaluator  │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │• Recall@K    │    │• Faithfulness│    │• ROUGE-1/2/L │          │
│  │• Precision@K │    │• Answer Rel  │    │• BLEU        │          │
│  │• MRR         │    │• Context Rel │    │• BERTScore   │          │
│  │• NDCG@K      │    │• Context Rec │    └──────────────┘          │
│  └──────────────┘    └──────────────┘                              │
│         │                   │                                       │
│         └───────────┬───────┘                                       │
│                     │                                               │
│                     ▼                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Fallback    │    │   Overall    │    │   Metrics    │          │
│  │  Evaluator   │───▶│   Score      │───▶│   Report     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---


## 评测执行流程

```
1. 数据准备阶段
   ├── 准备测试数据集
   │   ├── 可回答问题集 (question, ground_truth, relevant_docs)
   │   └── 无答案问题集 (unanswerable_question, expected_behavior)
   ├── 测试数据自动生成（推荐）
   │   ├── 从文档自动生成QA对
   │   ├── 数据增强策略
   │   └── 质量过滤
   ├── 格式校验
   └── 数据集划分: 训练集用于调参, 测试集用于最终评估

2. 检索评估阶段
   ├── 对每个query执行 KnowledgeBase.search(query, k=K)
   ├── 计算 Recall@K, Precision@K, MRR, NDCG@K
   └── 汇总检索指标 → Retrieval_Score

3. 生成评估阶段（一次LLM调用）
   ├── 获取检索返回的context
   ├── 获取生成的answer
   ├── 调用GenerationEvaluator.evaluate_generation_batch()并行评估四个指标
   └── 汇总生成质量指标 → Generation_Score

4. 端到端评估阶段
   ├── 对比generated_answer与ground_truth
   ├── 中文分词预处理 (jieba)
   ├── 计算ROUGE/BLEU分数
   ├── 计算BERTScore (语义层面)
   └── 汇总端到端指标 → E2E_Score

5. 兜底能力评估阶段
   ├── 对无答案问题执行RAG系统
   ├── 评估是否正确拒答
   ├── 评估幻觉率
   ├── 评估安全回复率
   └── 汇总兜底能力 → Fallback_Score

6. 综合评分与报告生成
   ├── 计算Overall_Score (加权公式)
   ├── 生成综合评测报告
   ├── 可视化展示
   └── 输出改进建议
```

---

## 评测报告输出格式

### Overall Score 加权公式

```
Overall_Score = w_retrieval * Retrieval_Score + w_generation * Generation_Score + w_e2e * E2E_Score + w_fallback * Fallback_Score

其中各维度得分的计算:

Retrieval_Score = 0.3 * Recall@K + 0.2 * Precision@K + 0.25 * MRR + 0.25 * NDCG@K

Generation_Score = 0.35 * Faithfulness + 0.30 * Answer_Relevancy + 0.20 * Context_Relevancy + 0.15 * Context_Recall

E2E_Score = 0.20 * ROUGE_1 + 0.15 * ROUGE_2 + 0.15 * ROUGE_L + 0.15 * BLEU + 0.35 * BERTScore

Fallback_Score = 0.50 * Reject_Accuracy + 0.30 * (1 - Hallucination_Rate) + 0.20 * Safe_Reply_Rate

推荐权重配置（可按业务场景调整）:
- 高检索质量要求: w_retrieval=0.40, w_generation=0.30, w_e2e=0.20, w_fallback=0.10
- 高生成质量要求: w_retrieval=0.25, w_generation=0.45, w_e2e=0.20, w_fallback=0.10
- 平衡配置: w_retrieval=0.30, w_generation=0.30, w_e2e=0.25, w_fallback=0.15
- 安全优先配置: w_retrieval=0.20, w_generation=0.25, w_e2e=0.15, w_fallback=0.40
```

### 评测报告JSON格式
