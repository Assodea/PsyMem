import os
import json
import re
import time
from openai import AzureOpenAI

# 初始化Azure OpenAI客户端
openai_client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)
openai_model = 'gpt-4o'

def load_structured_memory_files(base_path, client_id):
    """加载结构化的记忆文件 - 根据client_id加载对应的4个文件"""
    memories = []
    
    # 构建4个文件名
    file_names = [
        f"memory_{client_id}_1.json",
        f"memory_{client_id}_2.json",
        f"memory_{client_id}_3.json",
        f"memory_{client_id}_4.json"
    ]
    
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_name}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取双方记忆
                client_memory = data.get('client_memory', {})
                counselor_memory = data.get('counselor_memory', {})
                
                memories.append({
                    'filename': file_name,
                    'visit': data.get('visit', 0),
                    'timestamp': data.get('timestamp', ''),
                    'client_memory': {
                        'facts': client_memory.get('facts', []),
                        'suggestions': client_memory.get('suggestions', []),
                        'personality': client_memory.get('personality', ''),
                        'issues': client_memory.get('issues', [])
                    },
                    'counselor_memory': {
                        'facts': counselor_memory.get('facts', []),
                        'suggestions': counselor_memory.get('suggestions', []),
                        'personality': counselor_memory.get('personality', {}),
                        'issues': counselor_memory.get('issues', {})
                    }
                })
            print(f"已加载: {file_name} (第{data.get('visit', 0)}次咨询)")
        except Exception as e:
            print(f"加载失败 {file_name}: {e}")
    return memories

def format_memory_for_analysis(memories):
    """格式化记忆内容用于分析"""
    formatted_text = ""
    
    for i, mem in enumerate(memories, 1):
        # 确保issues是列表
        client_issues = mem['client_memory']['issues']
        if not isinstance(client_issues, list):
            client_issues = [client_issues] if client_issues else []
        
        # 安全地处理personality字段，确保它是字符串
        client_personality = mem['client_memory']['personality']
        if not isinstance(client_personality, str):
            if isinstance(client_personality, dict):
                client_personality = json.dumps(client_personality, ensure_ascii=False)
            elif isinstance(client_personality, list):
                client_personality = ", ".join(str(item) for item in client_personality)
            else:
                client_personality = str(client_personality) if client_personality else ""
        
        # 安全地处理counselor personality字段
        counselor_personality = mem['counselor_memory']['personality']
        if not isinstance(counselor_personality, dict):
            if isinstance(counselor_personality, str):
                try:
                    counselor_personality = json.loads(counselor_personality)
                except:
                    counselor_personality = {"personality": counselor_personality}
            else:
                counselor_personality = {}
        
        # 处理counselor issues字段（可能是字典或列表）
        counselor_issues = mem['counselor_memory']['issues']
        if isinstance(counselor_issues, dict):
            current_challenges = counselor_issues.get('current_challenges', [])
            progress_list = counselor_issues.get('progress', [])
        else:
            # 如果issues是列表，记录警告并置空
            print(f"警告: 咨询师记忆中的 issues 不是字典，而是 {type(counselor_issues)}，在文件 {mem['filename']} 中")
            current_challenges = []
            progress_list = []
        
        # 确保current_challenges和progress_list是列表
        if not isinstance(current_challenges, list):
            current_challenges = [current_challenges] if current_challenges else []
        if not isinstance(progress_list, list):
            progress_list = [progress_list] if progress_list else []
        
        formatted_text += f"""
【第{mem['visit']}次咨询 - {mem['filename']}】
时间: {mem['timestamp']}
{'-'*60}

1. 来访者记忆 (Client Memory):
   • 事实 (Facts):
     {chr(10).join(['     - ' + str(fact) for fact in mem['client_memory']['facts'][:5]])}
     {'     ...' if len(mem['client_memory']['facts']) > 5 else ''}
   
   • 建议 (Suggestions):
     {chr(10).join(['     - ' + str(suggestion) for suggestion in mem['client_memory']['suggestions'][:3]])}
     {'     ...' if len(mem['client_memory']['suggestions']) > 3 else ''}
   
   • 咨询师特点 (Personality): {client_personality[:100] if len(client_personality) > 100 else client_personality}{'...' if len(client_personality) > 100 else ''}
   
   • 核心问题 (Issues):
     {chr(10).join(['     - ' + str(issue) for issue in client_issues[:3]])}
     {'     ...' if len(client_issues) > 3 else ''}

2. 咨询师记忆 (Counselor Memory):
   • 事实 (Facts):
     {chr(10).join(['     - ' + str(fact) for fact in mem['counselor_memory']['facts'][:5]])}
     {'     ...' if len(mem['counselor_memory']['facts']) > 5 else ''}
   
   • 建议 (Suggestions):
     {chr(10).join(['     - ' + str(suggestion) for suggestion in mem['counselor_memory']['suggestions'][:3]])}
     {'     ...' if len(mem['counselor_memory']['suggestions']) > 3 else ''}
   
   • 来访者人格特点:
     {json.dumps(counselor_personality, ensure_ascii=False, indent=4)[:150] if counselor_personality else "无数据"}...
   
   • 治疗进展 (Issues):
     当前挑战: {chr(10).join(['        - ' + str(challenge) for challenge in current_challenges[:2]])}
     进展: {chr(10).join(['        - ' + str(progress) for progress in progress_list[:2]])}

{'='*80}
"""
    return formatted_text

def create_continuity_prompt(memories, client_id):
    """创建记忆延续性分析prompt"""
    
    formatted_memories = format_memory_for_analysis(memories)
    
    # 分析来访者的基本信息
    client_name = ""
    if memories:
        first_memory = memories[0]
        client_personality = first_memory.get('client_memory', {}).get('personality', '')
        
        if not isinstance(client_personality, str):
            if isinstance(client_personality, dict):
                client_personality = json.dumps(client_personality, ensure_ascii=False)
            elif isinstance(client_personality, list):
                client_personality = ", ".join(str(item) for item in client_personality)
            else:
                client_personality = str(client_personality) if client_personality else ""
        
        if "来访者" in client_personality:
            match = re.search(r'来访者[：:]\s*([^，。,;\n]+)', client_personality)
            if match:
                client_name = match.group(1)
    
    # 使用单引号定义模板字符串，所有JSON示例中的花括号已转义
    prompt_template = '''
## 任务：分析心理咨询记忆的延续性与连贯性（基于内容判断来访者一致性）

你是一名专业的心理咨询督导。你将收到**四次心理咨询的记忆记录**（每个文件包含来访者记忆和咨询师记忆）。  
**重要**：这些记录中的 `client_id` 可能相同，但实际可能是不同的人。你需要**通过记忆内容本身**判断这四次咨询是否来自**同一个真实的来访者**，并且是否构成**连续的治疗过程**。

### 判断标准（必须全部满足才视为“连贯”）：

1. **内容上为同一人**：比较四次咨询中来访者记忆的关键信息，不能存在根本性矛盾。包括但不限于：
   - 人口学信息（年龄、性别、职业、婚姻状况等）应一致或自然演变。
   - 核心生活背景（家庭成员、主要人际关系、重大生活事件）应连贯。
   - 主要困扰/症状（如焦虑、抑郁、人际关系问题）应有逻辑延续，而非完全无关。
   - 人格特质、价值观、语言风格应有可识别的一致性。
   - 如果出现无法解释的矛盾（如第一次说“我25岁单身”，第二次说“我40岁有两个孩子”），则判定为**不同人**。

2. **咨询序号连续**：咨询序号应为 1, 2, 3, 4 且没有缺失（或能合理推断为连续的四次）。

3. **治疗逻辑连贯**：四次咨询的内容应呈现合理的治疗进程（如探索→干预→评估），而不是跳跃、重复或倒退到完全无关的主题。

**如果以上任何一条不满足**，则判定为“不连贯”。此时你必须：
- 在所有延续性相关评分中给 **1分**（完全断裂）。
- 将 `strength_of_continuity` 设为 `"弱"`。
- 在 `evidence` 和 `detailed_analysis` 中明确指出具体的不连贯原因（例如：“第1次和第2次的来访者年龄/性别/核心问题矛盾，表明不是同一人”）。
- 在 `key_insights.biggest_discontinuity` 中描述最大的断裂点。

**如果三条都满足**，则正常进行延续性分析，评分范围 1-5 分（越高越连贯）。

## 记忆文件内容：

{formatted_memories}

## 分析要求（不满足连贯条件时仍需输出相同结构但给低分）

请从以下四个维度分析：

### 维度一：来访者记忆的自我延续性
分析来访者自身的记忆是否在四次咨询中发展和延续：
1. **情感状态延续**：情感体验、情绪波动、情感需求是否持续出现或有所变化？
2. **认知模式延续**：思维方式、信念系统、自我认知是否逐步深化？
3. **行动计划延续**：建议和实践计划是否具有连续性、递进性？
4. **成长轨迹**：是否呈现清晰的个人成长路径（从问题识别到行动计划）？

### 维度二：咨询师记忆的观察延续性
分析咨询师对来访者的观察是否在四次咨询中发展和延续：
1. **诊断一致性**：对来访者问题的识别是否一致且有深化？
2. **人格评估稳定性**：大五人格特质评估是否保持一致性？
3. **干预建议延续**：建议是否具有系统性、递进性？
4. **进展追踪**：是否清晰记录来访者的进步和挑战变化？

### 维度三：双方记忆的一致性
分析同一咨询中来访者与咨询师记忆的一致性：
1. **事实一致性**：双方对同一事件/感受的记录是否一致？
2. **问题认知一致性**：双方对核心问题的识别是否一致？
3. **建议对应性**：咨询师的建议是否回应了来访者的需求？
4. **情感基调一致性**：双方记录的情感基调是否匹配？

### 维度四：治疗过程的逻辑连贯性
分析四次咨询作为一个治疗过程的逻辑连贯性：
1. **治疗阶段递进**：是否呈现"问题评估→干预实施→效果评估→调整深化"的合理阶段？
2. **转折点识别**：是否存在明显的治疗转折点或突破点？
3. **脱落风险**：是否存在治疗停滞或倒退的迹象？
4. **治疗联盟强度**：双方记忆是否显示治疗关系的逐步深化？

## 输出格式（始终使用以下JSON结构）

{{
  "overall_continuity_assessment": {{
    "client_continuity_score": 1-5,
    "counselor_continuity_score": 1-5,
    "cross_session_consistency_score": 1-5,
    "therapeutic_coherence_score": 1-5,
    "overall_score": 1-5,
    "strength_of_continuity": "强/中/弱",
    "key_continuity_themes": ["主题1", "主题2", "主题3"]
  }},
  
  "client_memory_continuity": {{
    "emotional_state_continuity": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "evolution_pattern": "描述演变模式或'无'"
    }},
    "cognitive_pattern_continuity": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "evolution_pattern": "描述演变模式或'无'"
    }},
    "action_plan_continuity": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "evolution_pattern": "描述演变模式或'无'"
    }},
    "growth_trajectory": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "growth_stages": ["阶段1", "阶段2"]
    }}
  }},
  
  "counselor_memory_continuity": {{
    "diagnostic_consistency": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "diagnostic_deepening": true/false
    }},
    "personality_assessment_stability": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "trait_consistency": "描述一致性或'无'"
    }},
    "intervention_coherence": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "intervention_progression": "描述干预进展或'无'"
    }},
    "progress_tracking": {{
      "score": 1-5,
      "evidence": "具体证据或'不连贯：...'",
      "progress_pattern": "描述进步模式或'无'"
    }}
  }},
  
  "cross_memory_consistency": [
    {{
      "session": 1,
      "facts_consistency": {{
        "score": 1-5,
        "evidence": "具体匹配点",
        "discrepancies": ["不一致点1", "不一致点2"]
      }},
      "issues_alignment": {{
        "score": 1-5,
        "evidence": "具体匹配点",
        "discrepancies": ["不一致点1", "不一致点2"]
      }},
      "therapeutic_alliance_quality": "强/中/弱"
    }},
    {{
      "session": 2,
      "facts_consistency": {{}},
      "issues_alignment": {{}},
      "therapeutic_alliance_quality": "强/中/弱"
    }},
    {{
      "session": 3,
      "facts_consistency": {{}},
      "issues_alignment": {{}},
      "therapeutic_alliance_quality": "强/中/弱"
    }},
    {{
      "session": 4,
      "facts_consistency": {{}},
      "issues_alignment": {{}},
      "therapeutic_alliance_quality": "强/中/弱"
    }}
  ],
  
  "therapeutic_process_coherence": {{
    "logical_progression": {{
      "score": 1-5,
      "evidence": "具体证据",
      "treatment_stages": ["阶段1:问题评估", "阶段2:干预实施"]
    }},
    "turning_points": [
      {{
        "session": 1,
        "description": "转折点描述",
        "significance": "重要性说明"
      }}
    ],
    "stagnation_risks": [
      {{
        "session": 1,
        "description": "停滞风险描述",
        "severity": "高/中/低"
      }}
    ],
    "therapeutic_alliance_development": {{
      "score": 1-5,
      "evidence": "具体证据",
      "development_pattern": "描述发展模式"
    }}
  }},
  
  "key_insights": {{
    "most_continuous_element": "描述最具延续性的元素（若无则写'无'）",
    "biggest_discontinuity": "描述最大的不连贯点（如不同人、序号缺失、内容矛盾）",
    "therapeutic_effectiveness_indicator": "治疗有效性指标（若不连贯则写'无法评估'）",
    "recommendations_for_future_sessions": ["建议1", "建议2"]
  }},
  
  "detailed_analysis": "详细的文本分析，必须包含：基于内容判断是否为同一人、序号是否连续、内容是否逻辑连贯，以及任何断裂点的具体说明。请引用具体记忆内容作为证据。"
}}

## 评分标准

1分：完全断裂，无任何延续性（例如：内容显示为不同人、序号不连续、主题完全无关）  
2分：有明显断裂，但部分元素延续  
3分：基本连贯，但有明显不一致  
4分：良好连贯，偶有小问题  
5分：高度连贯，呈现清晰的治疗轨迹

## 注意事项

1. 请严格基于提供的记忆文件内容进行分析，不要编造信息。
2. 如果判定为不连贯（1分），请在 `detailed_analysis` 中明确列出矛盾点。
3. 即使不连贯，仍需填写所有字段，用合理占位符（如 `"无"`、`[]`）。
4. 输出必须是合法的 JSON 格式。

请开始分析。
'''

    # 填充模板并返回
    prompt = prompt_template.format(formatted_memories=formatted_memories)
    return prompt

def call_azure_openai(prompt):
    """调用Azure OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": """你是一名资深心理咨询督导，拥有丰富的临床经验和督导经验。你擅长：
1. 分析咨询过程记录的专业性和连贯性
2. 识别治疗过程中的延续性和断裂点
3. 评估来访者-咨询师双向记忆的一致性
4. 提供专业的治疗过程改进建议

请基于心理咨询的专业标准，客观、全面、细致地分析记忆延续性，并提供具体证据支持你的评估。"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        return result
        
    except Exception as e:
        print(f"调用API时出错: {e}")
        return None

def generate_detailed_report(parsed_results, client_id):
    """生成详细的评分报告文本"""
    report_lines = []
    
    report_lines.append("="*100)
    report_lines.append(f"来访者 {client_id} - 结构化记忆延续性详细评分分析")
    report_lines.append("="*100)
    
    # 1. 整体连续性评估
    report_lines.append(f"\n1. 整体连续性评估:")
    report_lines.append("-" * 50)
    overall = parsed_results.get("overall_continuity_assessment", {})
    
    if overall:
        for key, value in overall.items():
            if key.endswith("_score"):
                report_lines.append(f"   {key.replace('_', ' ').title()}: {value}/5")
            elif key == "key_continuity_themes" and isinstance(value, list):
                report_lines.append(f"   {key.replace('_', ' ').title()}:")
                for theme in value[:3]:
                    report_lines.append(f"     • {theme}")
            else:
                report_lines.append(f"   {key.replace('_', ' ').title()}: {value}")
    
    # 2. 来访者记忆连续性详细评分
    report_lines.append(f"\n2. 来访者记忆连续性详细评分:")
    report_lines.append("-" * 50)
    client_continuity = parsed_results.get("client_memory_continuity", {})
    
    if client_continuity:
        for category, details in client_continuity.items():
            if isinstance(details, dict) and 'score' in details:
                category_name = category.replace('_', ' ').title()
                report_lines.append(f"   {category_name}: {details['score']}/5")
                evidence = details.get('evidence', '')
                if evidence:
                    evidence_str = str(evidence)
                    if len(evidence_str) > 200:
                        evidence_str = evidence_str[:200] + "..."
                    report_lines.append(f"     证据摘要: {evidence_str}")
    
    # 3. 咨询师记忆连续性详细评分
    report_lines.append(f"\n3. 咨询师记忆连续性详细评分:")
    report_lines.append("-" * 50)
    counselor_continuity = parsed_results.get("counselor_memory_continuity", {})
    
    if counselor_continuity:
        for category, details in counselor_continuity.items():
            if isinstance(details, dict) and 'score' in details:
                category_name = category.replace('_', ' ').title()
                report_lines.append(f"   {category_name}: {details['score']}/5")
                evidence = details.get('evidence', '')
                if evidence:
                    evidence_str = str(evidence)
                    if len(evidence_str) > 200:
                        evidence_str = evidence_str[:200] + "..."
                    report_lines.append(f"     证据摘要: {evidence_str}")
    
    # 4. 双方记忆一致性详细评分（每次咨询）
    report_lines.append(f"\n4. 双方记忆一致性详细评分（按咨询次）:")
    report_lines.append("-" * 50)
    cross_consistency = parsed_results.get("cross_memory_consistency", [])
    
    if cross_consistency:
        for session_data in cross_consistency:
            session_num = session_data.get('session', 'N/A')
            report_lines.append(f"\n   第{session_num}次咨询:")
            
            facts_score = session_data.get('facts_consistency', {}).get('score', 'N/A')
            issues_score = session_data.get('issues_alignment', {}).get('score', 'N/A')
            alliance = session_data.get('therapeutic_alliance_quality', 'N/A')
            
            report_lines.append(f"     事实一致性: {facts_score}/5")
            report_lines.append(f"     问题认知一致性: {issues_score}/5")
            report_lines.append(f"     治疗联盟质量: {alliance}")
            
            discrepancies = session_data.get('facts_consistency', {}).get('discrepancies', [])
            if discrepancies:
                report_lines.append(f"     不一致点: {', '.join(str(d) for d in discrepancies[:2])}")
    
    # 5. 治疗过程连贯性详细评分
    report_lines.append(f"\n5. 治疗过程连贯性详细评分:")
    report_lines.append("-" * 50)
    therapeutic = parsed_results.get("therapeutic_process_coherence", {})
    
    if therapeutic:
        for category, details in therapeutic.items():
            if category not in ['turning_points', 'stagnation_risks'] and isinstance(details, dict) and 'score' in details:
                category_name = category.replace('_', ' ').title()
                report_lines.append(f"   {category_name}: {details['score']}/5")
        
        logical_progression = therapeutic.get('logical_progression', {})
        stages = logical_progression.get('treatment_stages', [])
        if stages:
            report_lines.append(f"\n   治疗阶段识别:")
            for i, stage in enumerate(stages, 1):
                report_lines.append(f"      阶段{i}: {stage}")
        
        turning_points = therapeutic.get('turning_points', [])
        if turning_points:
            report_lines.append(f"\n   治疗转折点:")
            for tp in turning_points[:2]:
                session = tp.get('session', 'N/A')
                desc = tp.get('description', 'N/A')
                significance = tp.get('significance', 'N/A')
                report_lines.append(f"      第{session}次咨询: {desc[:100]}{'...' if len(desc) > 100 else ''}")
                report_lines.append(f"      重要性: {significance[:100]}{'...' if len(significance) > 100 else ''}")
    
    # 6. 关键洞察
    report_lines.append(f"\n6. 关键洞察:")
    report_lines.append("-" * 50)
    insights = parsed_results.get("key_insights", {})
    
    if insights:
        most_continuous = insights.get('most_continuous_element', 'N/A')
        biggest_discontinuity = insights.get('biggest_discontinuity', 'N/A')
        
        most_continuous_str = str(most_continuous)
        biggest_discontinuity_str = str(biggest_discontinuity)
        
        report_lines.append(f"   最具延续性的元素: {most_continuous_str[:200] if len(most_continuous_str) > 200 else most_continuous_str}{'...' if len(most_continuous_str) > 200 else ''}")
        report_lines.append(f"   最大的不连贯点: {biggest_discontinuity_str[:200] if len(biggest_discontinuity_str) > 200 else biggest_discontinuity_str}{'...' if len(biggest_discontinuity_str) > 200 else ''}")
        
        effectiveness = insights.get('therapeutic_effectiveness_indicator', '')
        if effectiveness:
            report_lines.append(f"   治疗有效性指标: {str(effectiveness)[:200]}{'...' if len(str(effectiveness)) > 200 else ''}")
        
        recommendations = insights.get('recommendations_for_future_sessions', [])
        if recommendations:
            report_lines.append(f"   后续咨询建议:")
            for i, rec in enumerate(recommendations[:5], 1):
                report_lines.append(f"     {i}. {rec}")
    
    return "\n".join(report_lines)

def print_detailed_scores(parsed_results, client_id):
    """打印详细的评分分析到终端"""
    report_text = generate_detailed_report(parsed_results, client_id)
    print(report_text)

def extract_summary_from_results(parsed_results, client_id):
    """从单个分析结果中提取关键指标，用于整体汇总"""
    overall = parsed_results.get("overall_continuity_assessment", {})
    return {
        "client_id": client_id,
        "overall_score": overall.get("overall_score", None),
        "client_continuity_score": overall.get("client_continuity_score", None),
        "counselor_continuity_score": overall.get("counselor_continuity_score", None),
        "cross_session_consistency_score": overall.get("cross_session_consistency_score", None),
        "therapeutic_coherence_score": overall.get("therapeutic_coherence_score", None),
        "strength_of_continuity": overall.get("strength_of_continuity", None),
        "key_continuity_themes": overall.get("key_continuity_themes", []),
        "biggest_discontinuity": parsed_results.get("key_insights", {}).get("biggest_discontinuity", ""),
        "recommendations": parsed_results.get("key_insights", {}).get("recommendations_for_future_sessions", [])
    }

def save_and_summarize_results_with_collection(results, output_dir, client_id):
    """保存结果并返回摘要字典，用于整体汇总"""
    json_filename = f"structured_memory_continuity_analysis_{client_id}.json"
    txt_filename = f"structured_memory_continuity_summary_{client_id}.txt"
    
    json_path = os.path.join(output_dir, json_filename)
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        parsed_results = json.loads(results)
        
        # 保存JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_results, f, ensure_ascii=False, indent=2)
        print(f"\n结构化记忆分析结果已保存到: {json_path}")
        
        # 生成并保存文本报告
        report_text = generate_detailed_report(parsed_results, client_id)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"详细评分报告已保存到: {txt_path}")
        
        # 打印到终端
        print_detailed_scores(parsed_results, client_id)
        
        # 提取摘要
        summary = extract_summary_from_results(parsed_results, client_id)
        return summary
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(results)
        print(f"原始分析结果已保存到: {json_path}")
        return None
    except Exception as e:
        print(f"保存结果时出错: {e}")
        return None

def get_score_distribution(scores):
    """统计1-5分的分布"""
    dist = {1:0, 2:0, 3:0, 4:0, 5:0}
    for s in scores:
        if s in dist:
            dist[s] += 1
    return dist

def generate_overall_summary(all_results, failed_ids, all_requested_ids, output_dir):
    """生成整体汇总统计报告"""
    if not all_results:
        print("没有成功分析的结果，无法生成汇总报告。")
        return
    
    # 提取各分数列表
    overall_scores = [r["overall_score"] for r in all_results if r["overall_score"] is not None]
    client_scores = [r["client_continuity_score"] for r in all_results if r["client_continuity_score"] is not None]
    counselor_scores = [r["counselor_continuity_score"] for r in all_results if r["counselor_continuity_score"] is not None]
    cross_scores = [r["cross_session_consistency_score"] for r in all_results if r["cross_session_consistency_score"] is not None]
    therapeutic_scores = [r["therapeutic_coherence_score"] for r in all_results if r["therapeutic_coherence_score"] is not None]
    
    # 统计强度分布
    strength_counts = {"强": 0, "中": 0, "弱": 0}
    for r in all_results:
        strength = r.get("strength_of_continuity")
        if strength in strength_counts:
            strength_counts[strength] += 1
    
    # 统计所有主题词频
    theme_counter = {}
    for r in all_results:
        for theme in r.get("key_continuity_themes", []):
            theme_counter[theme] = theme_counter.get(theme, 0) + 1
    top_themes = sorted(theme_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 统计最常见的不连贯点（摘要）
    discontinuity_summary = {}
    for r in all_results:
        disc = r.get("biggest_discontinuity", "").strip()
        if disc and disc != "无":
            # 简化描述，取前30个字符作为分类（可选）
            key = disc[:30] + "..." if len(disc) > 30 else disc
            discontinuity_summary[key] = discontinuity_summary.get(key, 0) + 1
    
    # 统计最常见的建议
    rec_counter = {}
    for r in all_results:
        for rec in r.get("recommendations", [])[:3]:  # 只取前3条建议
            rec_counter[rec] = rec_counter.get(rec, 0) + 1
    top_recs = sorted(rec_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 计算平均值
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    summary = {
        "total_requested": len(all_requested_ids),
        "total_success": len(all_results),
        "total_failed": len(failed_ids),
        "failed_ids": failed_ids,
        "scores_summary": {
            "overall_score": {"avg": avg(overall_scores), "min": min(overall_scores) if overall_scores else None, "max": max(overall_scores) if overall_scores else None, "distribution": get_score_distribution(overall_scores)},
            "client_continuity_score": {"avg": avg(client_scores), "min": min(client_scores) if client_scores else None, "max": max(client_scores) if client_scores else None, "distribution": get_score_distribution(client_scores)},
            "counselor_continuity_score": {"avg": avg(counselor_scores), "min": min(counselor_scores) if counselor_scores else None, "max": max(counselor_scores) if counselor_scores else None, "distribution": get_score_distribution(counselor_scores)},
            "cross_session_consistency_score": {"avg": avg(cross_scores), "min": min(cross_scores) if cross_scores else None, "max": max(cross_scores) if cross_scores else None, "distribution": get_score_distribution(cross_scores)},
            "therapeutic_coherence_score": {"avg": avg(therapeutic_scores), "min": min(therapeutic_scores) if therapeutic_scores else None, "max": max(therapeutic_scores) if therapeutic_scores else None, "distribution": get_score_distribution(therapeutic_scores)},
        },
        "strength_of_continuity_counts": strength_counts,
        "top_continuity_themes": [{"theme": t, "count": c} for t, c in top_themes],
        "top_discontinuities": [{"description": d, "count": c} for d, c in discontinuity_summary.items()],
        "top_recommendations": [{"recommendation": r, "count": c} for r, c in top_recs],
    }
    
    # 保存JSON格式汇总
    json_summary_path = os.path.join(output_dir, "overall_summary.json")
    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n整体汇总报告(JSON)已保存: {json_summary_path}")
    
    # 生成可读的TXT汇总报告
    txt_summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(txt_summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("整体汇总统计报告 - 心理咨询记忆延续性分析\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"请求分析的来访者总数: {summary['total_requested']}\n")
        f.write(f"成功分析: {summary['total_success']}\n")
        f.write(f"失败: {summary['total_failed']}\n")
        if failed_ids:
            f.write(f"失败ID列表: {', '.join(failed_ids)}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("各维度平均得分\n")
        f.write("-"*40 + "\n")
        for key, val in summary["scores_summary"].items():
            f.write(f"{key}: 平均={val['avg']:.2f}, 最低={val['min']}, 最高={val['max']}\n")
            f.write(f"  分数分布: {val['distribution']}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("连续性强度分布\n")
        f.write("-"*40 + "\n")
        for strength, count in strength_counts.items():
            f.write(f"  {strength}: {count} 人\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("最常见的连续性主题 (Top 10)\n")
        f.write("-"*40 + "\n")
        for item in top_themes:
            f.write(f"  {item[0]}: {item[1]}次\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("最常见的不连贯点描述 (Top 10)\n")
        f.write("-"*40 + "\n")
        for item in summary["top_discontinuities"][:10]:
            f.write(f"  {item['description']}: {item['count']}次\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("最常见的后续建议 (Top 10)\n")
        f.write("-"*40 + "\n")
        for item in top_recs:
            f.write(f"  {item[0]}: {item[1]}次\n")
    
    print(f"可读汇总报告已保存: {txt_summary_path}")
    
    # 同时打印到终端
    print("\n" + "="*80)
    print("整体汇总统计结果")
    print("="*80)
    print(f"成功分析人数: {summary['total_success']} / {summary['total_requested']}")
    print(f"整体连续性平均分: {summary['scores_summary']['overall_score']['avg']:.2f}")
    print(f"来访者记忆连续性平均分: {summary['scores_summary']['client_continuity_score']['avg']:.2f}")
    print(f"咨询师记忆连续性平均分: {summary['scores_summary']['counselor_continuity_score']['avg']:.2f}")
    print(f"跨次一致性平均分: {summary['scores_summary']['cross_session_consistency_score']['avg']:.2f}")
    print(f"治疗过程连贯性平均分: {summary['scores_summary']['therapeutic_coherence_score']['avg']:.2f}")
    print("\n连续性强度分布:", strength_counts)
    print("\n最常见的连续性主题:")
    for item in top_themes[:5]:
        print(f"  - {item[0]}: {item[1]}次")
    print("\n最常见的建议:")
    for item in top_recs[:5]:
        print(f"  - {item[0]}: {item[1]}次")

def parse_id_range(input_str):
    """
    解析用户输入的ID范围，返回整数列表。
    支持格式：
      - 单个数字: "3" → [3]
      - 逗号分隔: "1,3,5" → [1,3,5]
      - 连字符范围: "1-5" → [1,2,3,4,5]
      - 混合: "1-3,5,7-9" → [1,2,3,5,7,8,9]
    """
    ids = set()
    # 按逗号分割
    parts = input_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            # 处理范围
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start > end:
                    start, end = end, start
                ids.update(range(start, end + 1))
            except ValueError:
                print(f"警告: 无法解析范围 '{part}'，已忽略")
        else:
            # 单个数字
            try:
                ids.add(int(part))
            except ValueError:
                print(f"警告: 无法解析数字 '{part}'，已忽略")
    return sorted(ids)

def main():
    # 根目录
    root_path = r"D:\VScodeProject\data"
    #python ourscontinue.py
    # 输出目录（固定）
    output_dir = os.path.join(root_path, "ablation_remove_txt_memorytxt")#D:\VScodeProject\data\2muti_dialogue2
    os.makedirs(output_dir, exist_ok=True)
    #D:\VScodeProject\data\ablation_remove_reD:\VScodeProject\data\ablation_remove_txt
    print("结构化记忆延续性分析系统 - 指定ID范围批量模式")
    print("-" * 50)
    
    # 输入根目录
    input_root = os.path.join(root_path, "ablation_remove_txt")
    if not os.path.exists(input_root):
        print(f"输入目录不存在: {input_root}")
        return
    
    # 获取用户指定的ID范围
    range_input = input("请输入要分析的来访者ID范围（例如 1-5 或 1,3,5）: ").strip()
    if not range_input:
        print("未输入任何ID，程序退出。")
        return
    
    client_ids = parse_id_range(range_input)
    if not client_ids:
        print("解析后没有有效的ID，程序退出。")
        return
    
    print(f"将分析以下来访者: {', '.join(str(i) for i in client_ids)}")
    
    # 用于整体汇总的列表
    all_results = []   # 存储每个成功来访者的摘要
    failed_ids = []    # 存储失败的ID
    
    # 循环处理每个client_id
    for client_id_int in client_ids:
        client_id = str(client_id_int)
        print(f"\n{'='*60}")
        print(f"开始处理来访者 {client_id}...")
        print(f"{'='*60}")
        
        input_dir = os.path.join(input_root, f"Consultation{client_id}")
        if not os.path.isdir(input_dir):
            print(f"目录不存在: {input_dir}，跳过")
            failed_ids.append(client_id)
            continue
        
        # 1. 加载记忆文件
        memories = load_structured_memory_files(input_dir, client_id)
        if len(memories) < 2:
            print(f"来访者 {client_id} 的有效记忆文件不足 (找到 {len(memories)} 个)，跳过")
            failed_ids.append(client_id)
            continue
        
        # 按咨询次数排序
        memories.sort(key=lambda x: x['visit'])
        
        # 2. 创建分析prompt
        print("\n正在生成分析提示...")
        prompt = create_continuity_prompt(memories, client_id)
        
        # 保存prompt（用于调试，可选）
        prompt_debug_path = os.path.join(output_dir, f"structured_analysis_prompt_{client_id}.txt")
        with open(prompt_debug_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Prompt已保存: {prompt_debug_path}")
        
        # 3. 调用Azure OpenAI
        print(f"\n正在调用Azure OpenAI进行分析来访者 {client_id}...")
        results = call_azure_openai(prompt)
        
        if results:
            # 4. 保存结果并收集摘要
            summary = save_and_summarize_results_with_collection(results, output_dir, client_id)
            if summary:
                all_results.append(summary)
            else:
                failed_ids.append(client_id)
        else:
            print(f"来访者 {client_id} 分析失败，请检查API配置")
            failed_ids.append(client_id)
    
    # ========== 生成整体汇总报告 ==========
    print("\n" + "="*80)
    print("批量分析完成，正在生成整体汇总统计报告...")
    print("="*80)
    
    generate_overall_summary(all_results, failed_ids, client_ids, output_dir)
    
    print("\n" + "="*60)
    print("所有指定ID分析完成！")
    print(f"输出文件位于: {output_dir}")
    print(f"成功分析: {len(all_results)} 人，失败: {len(failed_ids)} 人")

if __name__ == "__main__":
    main()