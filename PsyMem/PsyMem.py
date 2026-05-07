import os
import requests
import ujson
import time
from openai import AzureOpenAI
import traceback
#python PsyMem.py
client_url = 'http://127.0.0.1:8111/v1/chat/client/gpt-4o'
counselor_url = 'http://127.0.0.1:8112/v1/chat/counselor/gpt-4o'

# OpenAI客户端配置
openai_client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)
openai_model = 'gpt-4o'

# 判断回应内容是否有效（基础有效性校验）
def is_valid_response(content):
    if content is None:
        return False
    content = content.strip()
    if not content:
        return False
    end_puncts = ['。', '？', '！', '…', '.', '?', '!','...','。','”','！']
    has_valid_end = content[-1] in end_puncts
    no_truncation = not any(flag in content for flag in [ '未完', '等下', '接着说'])
    return has_valid_end and no_truncation

# 检查来访者回应是否符合要求：需包含1个具体细节和1个人格/背景相关的感受/行为
def check_client_response(content, role_card, turn):
    feedback = {
        "turn": turn,                  # 当前轮次
        "content": content,            # 原始回应内容
        "has_specific_detail": False,  # 是否包含具体细节（bool）
        "specific_detail": "",         # 提取的具体细节文本
        "has_personality_behavior": False,  # 是否包含人格/背景相关内容（bool）
        "personality_behavior": "",    # 提取的人格/背景相关文本
        "resistance_level": None,      # 从角色卡提取的阻抗程度（低/中/高）
        "need_rework": False,          # 是否需要重新生成回应（bool）
        "reason": ""                   # 返工原因（字符串）
    }
    
    # 检查是否包含具体细节（时间、场景、事件等）
    detail_keywords = ["昨天", "今天", "上周", "在", "和", "那天", "时候", "一次", "比如", "例如"]
    if any(kw in content for kw in detail_keywords) or len(content.split("。")[0]) > 20:
        feedback["has_specific_detail"] = True
        feedback["specific_detail"] = content.split("。")[0].strip()

    # 提取角色卡中的背景线索（用于匹配人格/背景相关内容）
    background_clues = []
    if "职业：" in role_card:
        background_clues.append(role_card.split("职业：")[1].split("，")[0].strip())
    if "婚恋状态：" in role_card:
        background_clues.append(role_card.split("婚恋状态：")[1].split("，")[0].strip())
    if "阻抗程度：" in role_card:
        resistance_label = role_card.split("阻抗程度：")[1].split("，")[0].strip()
        feedback["resistance_level"] = resistance_label
        background_clues.append(resistance_label)

    # 优先匹配角色卡中的背景线索
    for clue in background_clues:
        if clue in content.lower():
            feedback["has_personality_behavior"] = True
            feedback["personality_behavior"] = [c for c in background_clues if c in content.lower()]
            break
    
    # 若未匹配到背景线索，检查是否有明确的感受描述
    if not feedback["has_personality_behavior"]:
        emotion_words = ["烦躁", "难过", "不想说", "不愿意", "觉得", "感觉", "抵触", "犹豫"]
        if any(emotion in content for emotion in emotion_words):
            feedback["has_personality_behavior"] = True
            feedback["personality_behavior"] = [e for e in emotion_words if e in content]

    # 判断是否需要返工（缺少具体细节或人格/背景相关内容则需返工）
    if not feedback["has_specific_detail"] or not feedback["has_personality_behavior"]:
        feedback["need_rework"] = True
        reasons = []
        if not feedback["has_specific_detail"]:
            reasons.append("缺少具体细节（如具体事件、场景、时间）")
        if not feedback["has_personality_behavior"]:
            reasons.append("缺少与人格/背景相关的感受或行为（如职业相关情绪、阻抗表现）")
        feedback["reason"] = "; ".join(reasons)

    return feedback

# 自然结束对话的关键词列表
natural_end_keywords = [
    '先聊到这里', '今天就到这', '再见', '我们下次见','明天见','晚安','下次再聊'    
    '先这样吧', '谢谢，不聊了', '结束今天的对话', '下次咨询再说',
    '今天的时间差不多了', '我们下次再深入聊', '先到这里休息一下',
    '有需要可以再来找我', '今天先到这，下次见', '今天的咨询就到这里',
    '我会随时来找你','谢谢你今天的陪伴','一起加油','随时来找我','祝你'
]

# 检查内容末尾是否包含结束关键词
def is_end_keyword_in_tail(content, keywords, tail_length=50):
    if content is None:
        return False
    content_tail = content[-tail_length:].strip()
    return any(kw in content_tail for kw in keywords)

# 从TXT结构化内容中提取来访者的主诉问题
def extract_chief_problem(profile_detail: str) -> str:
    cleaned_detail = profile_detail.replace(' ', '').replace('\n', '').strip()
    
    start_keys = ["当前主要心理困扰：", "当前主要心理困扰:"]
    end_keys = ["-心理咨询相关：", "-心理咨询相关:", "###二、童年时期"]
    matched_key = None
    for key in start_keys:
        if key in cleaned_detail:
            matched_key = key
            break
    if not matched_key:
        print("[提取警告] TXT中未找到'当前主要心理困扰'！")
        return "未明确具体困扰，需结合个人经历进一步沟通"
    
    # 截取起始关键词后的内容（即主诉问题部分）
    problem_content = cleaned_detail.split(matched_key)[-1]
    for end_key in end_keys:
        if end_key in problem_content:
            problem_content = problem_content.split(end_key)[0].strip()
            break
    
    return problem_content if problem_content and len(problem_content) >=5 else "未明确具体困扰，需结合个人经历进一步沟通"

# 生成首次访问的角色卡（从TXT提取主诉）
def get_initial_role_card(profile_detail: str):
    chief_problem = extract_chief_problem(profile_detail)
    return f'''{profile_detail}
首次来访主诉问题：{chief_problem}'''

# 后续访问更新主诉的函数（结合TXT原始内容和历史对话）
def update_chief_problem(initial_role_card, previous_session, profile_detail):
    """
    结合TXT原始背景信息和历史对话更新主诉
    参数：
        initial_role_card: 初始角色卡（保留原始信息）
        previous_session: 上一次访问的对话历史
        profile_detail: TXT原始内容（用于提取固定背景）
    返回：
        updated_problem: 更新后的主诉问题
    """
    # 从TXT提取固定背景信息（职业、婚恋状态等）
    background_info = []
    if "职业：" in profile_detail:
        background_info.append(f"职业：{profile_detail.split('职业：')[1].split('，')[0].strip()}")
    if "婚恋状态：" in profile_detail:
        background_info.append(f"婚恋状态：{profile_detail.split('婚恋状态：')[1].split('，')[0].strip()}")
    
    # 从历史对话提取最近进展（最后3轮）
    if len(previous_session) < 3:
        last_content = [msg['content'] for msg in previous_session]
    else:
        last_content = [msg['content'] for msg in previous_session[-3:]]
    recent_progress = f"近期沟通提到：{last_content[-1][:30]}..." if last_content else "上次沟通未提及具体内容"
    
    # 结合背景和新进展生成更新后主诉
    return f"后续访问主诉（{'; '.join(background_info)}）：{recent_progress}"

# 生成后续访问的角色卡（保留初始信息+更新主诉）
def get_updated_role_card(initial_role_card, previous_session, visit, profile_detail):
    updated_problem = update_chief_problem(initial_role_card, previous_session, profile_detail)
    return f'''{initial_role_card}
第{visit}次访问主诉更新：{updated_problem}'''

# 安全的API请求函数，处理JSON解析错误
def safe_api_request(url, json_data, max_retries=3):
    """安全的API请求，处理JSON解析错误"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, timeout=120)
            response.raise_for_status()  # 检查HTTP状态码
            
            # 尝试解析JSON
            try:
                result = response.json()
                return result
            except ValueError as json_err:
                print(f"第{attempt+1}次尝试：JSON解析失败: {json_err}")
                print(f"响应内容: {response.text[:200]}")
                
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回默认值
                    return {"item": {"content": "系统错误，请重试", "role": "system"}}
                
                # 等待后重试
                time.sleep(1)
                continue
                
        except requests.exceptions.RequestException as req_err:
            print(f"第{attempt+1}次尝试：请求失败: {req_err}")
            
            if attempt == max_retries - 1:
                return {"item": {"content": "网络错误，请稍后重试", "role": "system"}}
            
            time.sleep(2)  # 等待更长时间后重试
    
    return {"item": {"content": "请求失败", "role": "system"}}

# ==================== 新增：咨询师侧记忆注入函数 ====================
def inject_memory_to_counselor(session, previous_memory):
    """
    将上一次的记忆总结注入到咨询师的会话中（作为 system 消息），
    并指示模型不要显式提及“上次”等字眼。
    返回新的 session 列表（不修改原始 session）。
    """
    if not previous_memory or not isinstance(previous_memory, dict):
        return session  # 无记忆时直接返回原 session
    
    mem = previous_memory.get('counselor_memory', {})
    facts = mem.get('facts', [])
    issues = mem.get('issues', '')
    personality = mem.get('personality', '')
    
    memory_text = ""
    if facts:
        if isinstance(facts, list):
            facts_str = "；".join([str(f) for f in facts if f])
        else:
            facts_str = str(facts)
        memory_text += f"来访者曾经历：{facts_str}。"
    if issues and issues != "无":
        memory_text += f"核心困扰：{issues}。"
    if personality and personality != "无":
        memory_text += f"来访者特点：{personality}。"
    
    if not memory_text:
        return session
    
    # 构建 system 消息，明确要求不提及“上次”
    instruction = "请利用以上信息来理解来访者的背景和经历。"
    memory_message = {"role": "system", "content": memory_text + instruction}
    
    # 返回新的 session 列表（原 session 不变，仅用于本次请求）
    return [memory_message] + session
# ==================== 新增结束 ====================

# 使用大模型生成单视角记忆总结
def generate_memory_summary_with_llm(session, previous_memory, perspective, role_card, visit):
    """
    使用大模型生成单视角记忆总结
    参数：
        session: 当前对话历史
        previous_memory: 上一次的记忆总结（字典格式）
        perspective: 视角，'client' 或 'counselor'
        role_card: 角色卡信息
        visit: 当前访问次数
    返回：
        记忆总结字典
    """
    
    # 构建对话历史文本
    dialog_text = ""
    for msg in session:
        role_name = "来访者" if msg['role'] == 'client' else "咨询师"
        dialog_text += f"{role_name}: {msg['content']}\n"
    
    # 安全地构建上一次记忆总结文本
    previous_memory_text = ""
    if previous_memory and isinstance(previous_memory, dict):
        try:
            if perspective == 'client':
                prev_mem = previous_memory.get('client_memory', {})
            else:
                prev_mem = previous_memory.get('counselor_memory', {})
            
            if prev_mem and isinstance(prev_mem, dict):
                # 安全处理facts和suggestions
                facts_list = prev_mem.get('facts', [])
                suggestions_list = prev_mem.get('suggestions', [])
                
                # 确保每个元素都是字符串
                safe_facts = []
                for item in facts_list:
                    if isinstance(item, dict):
                        # 如果是字典，转换为字符串
                        safe_facts.append(str(item))
                    else:
                        safe_facts.append(str(item))
                
                safe_suggestions = []
                for item in suggestions_list:
                    if isinstance(item, dict):
                        safe_suggestions.append(str(item))
                    else:
                        safe_suggestions.append(str(item))
                
                # 构建记忆总结文本
                facts_text = ', '.join(safe_facts) if safe_facts else '无'
                suggestions_text = ', '.join(safe_suggestions) if safe_suggestions else '无'
                personality_text = str(prev_mem.get('personality', '无'))
                issues_text = str(prev_mem.get('issues', '无'))
                
                previous_memory_text = f"""上一次咨询的记忆总结：
- 重要事实: {facts_text}
- 建议记录: {suggestions_text}
- 对方特点: {personality_text}
- 核心问题: {issues_text}
"""
        except Exception as e:
            print(f"处理上一次记忆总结时出错: {e}")
            previous_memory_text = "上一次记忆总结格式异常"
    
    # 构建系统提示词
    if perspective == 'client':
        system_prompt = f"""你是一位心理咨询来访者，刚刚完成了第{visit}次心理咨询。请基于本次咨询的对话内容，并结合上一次咨询的记忆总结，生成你的个人记忆总结。

你的角色信息：
{role_card}

生成记忆总结时，请按照以下结构化格式思考：
1. 重要事实：记录本次咨询中讨论的重要事件、时间和具体细节
2. 咨询师建议：记录咨询师给出的重要建议和指导
3. 咨询师特点：总结咨询师的风格、态度和专业性
4. 核心问题：总结你对自己问题的认识进展

请用中文输出JSON格式，包含以下键：facts（事实列表）、suggestions（建议列表）、personality（咨询师特点描述）、issues（主诉问题描述）。"""
    else:
        system_prompt = f"""你是一位心理咨询师，刚刚完成了与来访者的第{visit}次咨询。请基于本次咨询的对话内容，并结合上一次咨询的记忆总结，生成你的专业记忆总结。

来访者角色信息：
{role_card}

生成记忆总结时，请按照以下结构化格式思考：
1. 重要事实：记录来访者提及的重要事件、时间和具体细节
2. 建议记录：记录你给出的重要建议和干预措施
3. 来访者特点：总结来访者的性格特点、阻抗程度和改变意愿
4. 咨询进展：总结来访者问题的演变和治疗进展

请用中文输出JSON格式，包含以下键：facts（事实列表）、suggestions（建议列表）、personality（来访者特点描述）、progress（咨询进展）。"""
    
    # 构建用户消息
    user_message = f"""本次咨询对话内容：
{dialog_text}

{previous_memory_text}

请基于以上对话内容和上一次的记忆总结，生成第{visit}次咨询的{'来访者' if perspective == 'client' else '咨询师'}视角记忆总结。

请确保总结：
1. 聚焦本次咨询的新内容
2. 适当参考上一次的记忆总结
3. 保持专业性和客观性
4. 突出重要进展和变化

请直接输出JSON格式，不要有其他解释文本。"""
    
    try:
        # 调用Azure OpenAI API
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # 解析JSON响应
        memory_summary = ujson.loads(response.choices[0].message.content)
        
        # 确保返回的字典包含必要的键
        required_keys = ["facts", "suggestions", "personality", "issues"]
        for key in required_keys:
            if key not in memory_summary:
                memory_summary[key] = "" if key != "facts" and key != "suggestions" else []
        
        # 确保facts和suggestions是列表
        if isinstance(memory_summary["facts"], str):
            memory_summary["facts"] = [memory_summary["facts"]] if memory_summary["facts"] else []
        elif not isinstance(memory_summary["facts"], list):
            memory_summary["facts"] = []
        
        if isinstance(memory_summary["suggestions"], str):
            memory_summary["suggestions"] = [memory_summary["suggestions"]] if memory_summary["suggestions"] else []
        elif not isinstance(memory_summary["suggestions"], list):
            memory_summary["suggestions"] = []
        
        return memory_summary
        
    except Exception as e:
        print(f"大模型生成记忆总结失败: {e}")
        traceback.print_exc()
        # 返回默认的空总结
        return {
            "facts": [f"第{visit}次咨询对话总结生成失败"],
            "suggestions": [],
            "personality": "无法生成总结",
            "issues": f"第{visit}次咨询总结错误: {str(e)[:50]}"
        }

# 保存双视角记忆总结
def save_memory_summaries(idx, visit, session, role_card, saved_dir, previous_memory=None):
    """
    生成并保存双视角记忆总结
    参数：
        idx: 来访者索引
        visit: 当前访问次数
        session: 当前对话历史
        role_card: 角色卡信息
        saved_dir: 保存目录
        previous_memory: 上一次的记忆总结（字典格式）
    返回：
        当前记忆总结字典
    """
    
    # 生成来访者视角记忆总结
    client_memory = generate_memory_summary_with_llm(
        session=session,
        previous_memory=previous_memory,
        perspective='client',
        role_card=role_card,
        visit=visit
    )
    
    # 生成咨询师视角记忆总结
    counselor_memory = generate_memory_summary_with_llm(
        session=session,
        previous_memory=previous_memory,
        perspective='counselor',
        role_card=role_card,
        visit=visit
    )
    
    # 构建完整的记忆总结
    memory = {
        "client_memory": client_memory,
        "counselor_memory": counselor_memory,
        "visit": visit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存到文件
    memory_file = os.path.join(saved_dir, f'memory_{idx}_{visit}.json')
    with open(memory_file, 'w', encoding='utf-8') as f:
        ujson.dump(memory, f, ensure_ascii=False, indent=2)
    
    print(f"第 {idx} 位来访者第 {visit} 次访问记忆总结已保存到 {memory_file}")
    return memory

# 主程序配置
DEBUG_MODE = True #False
USE_100TURN_PROMPT = True
saved_dir = 'longdialogue'
os.makedirs(saved_dir, exist_ok=True)
target_indices = [0]  # 5位来访者
visit_count = 4  # 每位4次咨询
MAX_RETRY_TIMES = 10  # 最大连续返工阈值（超过后停止重试）
#python PsyMem.py
# ====== 预热两个 API（仅触发一次，避免冷启动空响应）======
print("预热来访者 API...")
try:
    requests.post(client_url, json={
        'session': [{'role': 'client', 'content': '你好'}],
        'role_card': 'test'
    }, timeout=120)
    print("来访者 API 预热完成")
except Exception as e:
    print(f"来访者 API 预热失败（不影响后续）: {e}")

print("预热咨询师 API...")
try:
    requests.post(counselor_url, json={
        'session': [{'role': 'client', 'content': '你好'}]
    }, timeout=120)
    print("咨询师 API 预热完成")
except Exception as e:
    print(f"咨询师 API 预热失败（不影响后续）: {e}")
# =====================================================

for idx in target_indices:
    # 读取基础画像（所有访问共用的TXT内容）
    try:
        txt_file_path = rf'D:\VScodeProject\inter\interactive-agents-main\oursRoleCard\TXT{idx}.txt'
        #D:\VScodeProject\inter\interactive-agents-main\oursRoleCard\TXT{idx}.txt
        with open(txt_file_path, 'r', encoding='utf-8') as f2:
            profile_detail = f2.read().strip()
        
        # 初始化初始角色卡（仅首次访问时生成）
        initial_role_card = get_initial_role_card(profile_detail)

        # 调试模式：仅保存首次访问的初始角色卡
        if DEBUG_MODE:
            debug_dir = 'debug_role_cards'
            os.makedirs(debug_dir, exist_ok=True)
            debug_file_path = os.path.join(debug_dir, f'initial_role_card_{idx}.txt')
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write("--- 初始角色卡（首次访问） ---\n\n")
                f.write(initial_role_card)
            print(f"[调试模式] 首次访问角色卡已保存到: {debug_file_path}")

    except Exception as e:
        print(f"读取来访者{idx}画像失败: {e}")
        continue

    # 存储上一次访问的会话和记忆总结
    previous_session = None
    previous_memory = None  # 用于存储上一次的记忆总结

    for visit in range(1, visit_count + 1):
        print(f"开始生成第 {idx} 位来访者的第 {visit} 次访问对话...")
        try:
            # 确定当前访问的角色卡
            if visit == 1:
                current_role_card = initial_role_card
            else:
                # 从上次访问的会话文件加载历史
                prev_session_path = os.path.join(saved_dir, f'{idx}_{visit-1}.json')
                if os.path.exists(prev_session_path):
                    try:
                        with open(prev_session_path, 'r', encoding='utf-8') as f:
                            previous_session = ujson.load(f)
                    except Exception as e:
                        print(f"加载上一次会话失败: {e}")
                        previous_session = []
                else:
                    print(f"警告：未找到上一次访问记录{prev_session_path}，使用初始角色卡")
                    previous_session = []
                # 生成更新后的角色卡（传入TXT原始内容）
                current_role_card = get_updated_role_card(initial_role_card, previous_session, visit, profile_detail)
                
                # 加载上一次的记忆总结（如果是第二次及以后的访问）
                prev_memory_path = os.path.join(saved_dir, f'memory_{idx}_{visit-1}.json')
                if os.path.exists(prev_memory_path):
                    try:
                        with open(prev_memory_path, 'r', encoding='utf-8') as f:
                            previous_memory = ujson.load(f)
                        print(f"已加载第 {idx} 位来访者第 {visit-1} 次访问的记忆总结")
                    except Exception as e:
                        print(f"加载上一次记忆总结失败: {e}")
                        previous_memory = None

            # 初始化本次访问会话
            if visit == 1:
                session = [{'role': 'client', 'content': '你好，我最近心里有点不舒服，想跟你聊聊。'}]
            else:
                # 后续访问承接上次话题
                session = [{'role': 'client', 'content': '你好，我又来了，想继续聊聊上次的情况。'}]
            print(f"第 {idx} 位来访者第 {visit} 次访问开始: {session[0]}")

            # ====== 咨询师首次回应（修改：注入记忆） ======
            session_with_memory = inject_memory_to_counselor(session, previous_memory)
            counselor_response = safe_api_request(counselor_url, {'session': session_with_memory})
            counselor_item = counselor_response.get('item', {})
            while counselor_item.get('content') is None or not is_valid_response(counselor_item['content']):
                content = counselor_item.get('content')
                if content is None or (isinstance(content, str) and content.strip() == ''):
                    print("咨询师回应为空，重新请求...")
                else:
                    print(f"咨询师回应基础无效，内容: {content}，重新请求...")
                session_with_memory = inject_memory_to_counselor(session, previous_memory)
                counselor_response = safe_api_request(counselor_url, {'session': session_with_memory})
                counselor_item = counselor_response.get('item', {})
            session.append(counselor_item)

            client_feedback = []
            for turn in range(1, 150):  # 最大轮次限制
                print(f"第 {idx} 位来访者第 {visit} 次访问第 {turn} 轮...")

                # 来访者回应生成与校验 - 使用安全的API请求
                client_response = safe_api_request(
                    client_url, 
                    {'session': session, 'role_card': current_role_card}
                )
                client_item = client_response.get('item', {})
    
                retry_count = 0  # 初始化连续返工计数器
                while True:
                    retry_count += 1  # 每次循环计数+1
                    
                    # 1. 基础有效性校验（非空+完整结尾）
                    content = client_item.get('content')
                    valid_base = content is not None and is_valid_response(content)
                    if not valid_base:
                        # 区分空内容与无效内容，打印不同日志
                        if content is None or (isinstance(content, str) and content.strip() == ''):
                            print("来访者回应为空，重新请求...")
                        else:
                            print(f"来访者回应基础无效，内容: {content}，重新请求...")
                        
                        # 重新请求（逻辑与原代码完全一致，无上限、无兜底替换）
                        client_response = safe_api_request(
                            client_url,
                            {'session': session, 'role_card': current_role_card}
                        )
                        client_item = client_response.get('item', {})
                        continue

                    # 2. 质量检查（是否包含具体细节和人格/背景相关内容）
                    client_content = client_item['content']
                    feedback = check_client_response(client_content, current_role_card, turn)
                    
                    if feedback["need_rework"]:
                        # 检查是否达到最大重试阈值
                        if retry_count >= MAX_RETRY_TIMES:
                            print(f"第 {idx} 位来访者第 {visit} 次访问第 {turn} 轮达到最大重试次数（{MAX_RETRY_TIMES}次），停止返工继续推进")
                            # 即使未通过质量检查，也记录反馈并退出循环
                            client_feedback.append({**feedback, "force_continue": True})
                            break
                        
                        # 未达到阈值，继续重试
                        print(f"第 {idx} 位来访者第 {visit} 次访问第 {turn} 轮需返工（{retry_count}/{MAX_RETRY_TIMES}）：{feedback['reason']}")
                        client_response = safe_api_request(
                            client_url, 
                            {'session': session, 'role_card': current_role_card}
                        )
                        client_item = client_response.get('item', {})
                    else:
                        # 检查通过，记录反馈并退出循环
                        print(f"第 {idx} 位来访者第 {visit} 次访问第 {turn} 轮检查通过")
                        client_feedback.append(feedback)
                        break

                # 确保回应非空（最后兜底）
                while client_item.get('content') is None or not is_valid_response(client_item['content']):
                    print("来访者回应兜底检查失败，重新请求...")
                    client_response = safe_api_request(
                        client_url, 
                        {'session': session, 'role_card': current_role_card}
                    )
                    client_item = client_response.get('item', {})
                
                session.append(client_item)

                # 检查来访者是否主动结束
                client_content = client_item['content']
                if is_end_keyword_in_tail(client_content, natural_end_keywords) and is_valid_response(client_content):
                    print(f"第 {idx} 位来访者第 {visit} 次访问因来访者主动结束于第{turn}轮")
                    break
                
                # 咨询师回应
                if USE_100TURN_PROMPT and turn == 99:
                    counselor_content = "我们今天已经聊了不少内容了，感觉也涉及到很多方面了。你看是想继续再多说一会儿，还是今天先到这里，下次我们再接着这个话题聊呢？"
                    counselor_item = {'role': 'counselor', 'content': counselor_content}
                else:
                    # ====== 修改：咨询师普通回应（注入记忆） ======
                    session_with_memory = inject_memory_to_counselor(session, previous_memory)
                    counselor_response = safe_api_request(counselor_url, {'session': session_with_memory})
                    counselor_item = counselor_response.get('item', {})
                    while counselor_item.get('content') is None or not is_valid_response(counselor_item['content']):
                        content = counselor_item.get('content')
                        if content is None or (isinstance(content, str) and content.strip() == ''):
                            print("咨询师回应为空，重新请求...")
                        else:
                            print(f"咨询师回应基础无效，内容: {content}，重新请求...")
                        session_with_memory = inject_memory_to_counselor(session, previous_memory)
                        counselor_response = safe_api_request(counselor_url, {'session': session_with_memory})
                        counselor_item = counselor_response.get('item', {})
                session.append(counselor_item)

                # 每轮保存会话
                session_file = os.path.join(saved_dir, f'{idx}_{visit}.json')
                with open(session_file, 'w', encoding='utf-8') as f3:
                    ujson.dump(session, f3, ensure_ascii=False, indent=2)

                # 检查咨询师是否提议结束
                counselor_content = counselor_item['content']
                if is_end_keyword_in_tail(counselor_content, natural_end_keywords) and is_valid_response(counselor_content):
                    print(f"第 {idx} 位来访者第 {visit} 次访问因咨询师提议结束于第{turn}轮")
                    break

            # 保存最终会话
            session_file = os.path.join(saved_dir, f'{idx}_{visit}.json')
            with open(session_file, 'w', encoding='utf-8') as f3:
                ujson.dump(session, f3, ensure_ascii=False, indent=2)

            # 保存来访者反馈
            feedback_file = os.path.join(saved_dir, f're{idx}_{visit}.json')
            with open(feedback_file, 'w', encoding='utf-8') as f:
                ujson.dump(client_feedback, f, ensure_ascii=False, indent=2)
            print(f"第 {idx} 位来访者第 {visit} 次访问反馈已保存至 {feedback_file}")

            # 生成并保存双视角记忆总结
            try:
                current_memory = save_memory_summaries(
                    idx=idx,
                    visit=visit,
                    session=session,
                    role_card=current_role_card,
                    saved_dir=saved_dir,
                    previous_memory=previous_memory
                )
            except Exception as e:
                print(f"生成记忆总结失败: {e}")
                traceback.print_exc()
                current_memory = {
                    "client_memory": {"facts": [], "suggestions": [], "personality": "生成失败", "issues": str(e)},
                    "counselor_memory": {"facts": [], "suggestions": [], "personality": "生成失败", "issues": str(e)},
                    "visit": visit,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # 更新上一次会话和记忆总结为当前（用于下一次访问）
            previous_session = session
            previous_memory = current_memory

        except Exception as e:
            print(f"生成第 {idx} 位来访者第 {visit} 次访问时出错: {e}")
            traceback.print_exc()
            continue

print(f"5位来访者各{visit_count}次访问的对话生成完成！")

#对话内容（{idx}_{visit}.json）
#来访者反馈（re{idx}_{visit}.json）
#每次咨询主诉（problem_{idx}_{visit}.txt）不保存
#双视角记忆总结（memory_{idx}_{visit}.json）