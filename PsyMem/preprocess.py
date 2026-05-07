import os
from pathlib import Path
from openai import AzureOpenAI
#python preprocess.py
# -------------------------- 全局配置（纯TXT输入输出）--------------------------
# Azure OpenAI 客户端配置（依赖：pip install openai==1.13.3）
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
openai_model = 'gpt-4o'

# ======================== 目录 ========================
# 新输入/输出目录（使用正斜杠，Path对象自动兼容Windows）
INPUT_DIR = Path(r"D:/VScodeProject/inter/interactive-agents-main/raw_role_card/train")
OUTPUT_DIR = Path(r"D:/VScodeProject/inter/interactive-agents-main/oursRoleCard")

# 文件名映射（输入：1.txt → 输出：TXT1.txt）
def get_input_path(idx: int) -> Path:
    return INPUT_DIR / f"{idx}.txt"

def get_output_path(idx: int) -> Path:
    return OUTPUT_DIR / f"TXT{idx}.txt"
# ========================  ========================

# 默认处理前5个文件，可修改此列表
target_indices = [0]
#python preprocess.py
# 通用精神分析Prompt模板
PROMPT_TEMPLATE = """基于精神分析理论，结合以下用户核心信息，生成结构严谨、逻辑闭环的人物小传（1500-2000字），必须严格遵循以下固定结构与格式要求，不得遗漏任何模块，不得打乱顺序：

一、核心档案（强制列表格式，字段不可缺失）
- 基础信息：性别、年龄、教育程度、职业、婚恋情况
- 核心心理特征：大五人格关键特质（提炼3-4个核心点，不罗列原文）
- 家庭关系核心矛盾：基于原文提炼1-2个关键冲突/缺失（如“原生家庭情感疏离”“性别偏向导致的被忽视”）
- 当前主要心理困扰：明确影响其生活/学习/关系的核心问题（如“亲密关系丧失后的孤独感”“抑郁与自残倾向”）
- 心理咨询相关：阻抗程度、求助意愿

二、童年时期（0-12岁：人格基础奠基期）
1. 早期客体关系（核心分析维度）
- 主要依恋对象（如父母、祖辈）及互动模式（补充合理场景细节，贴合原文家庭关系描述）
- 安全感建立状态（基于家庭环境推导，如“稳定依恋→安全感充足”“矛盾型互动→安全感缺失”）
2. 人格特质雏形（因果对应）
- 结合原文大五人格，分析童年经历如何催生特质雏形（如“父母高要求→尽责性萌芽”“情感回应不足→外向性偏低”）
3. 潜意识核心信念形成（精神分析核心）
- 推导童年时期形成的隐性信念（如“只有付出才能获得认可”“我是不被重视的”），需与后续心理困扰呼应

三、青年时期（13-当前年龄前：人格冲突显现期）
1. 关键生活事件（强制2-3个核心事件）
- 事件需贴合原文背景（如学业选择、亲密关系建立/破裂、家庭变动），补充具体场景细节
- 说明事件对人格的“强化/冲突”作用（如“亲密关系破裂→强化神经质特质”“出国留学→触发独立与依赖的冲突”）
2. 人格特质固化与冲突（逻辑闭环）
- 分析大五人格特质如何在青年阶段固化（如“高尽责性→自学两年提升自我价值”）
- 明确特质间的核心冲突（如“高宜人性与低信任感的矛盾”“开放性与对改变的恐惧冲突”）
3. 防御机制形成（精神分析关键视角）
- 提炼1-2种核心防御机制（如“回避型防御→通过独处逃避家庭矛盾”“讨好型防御→以付出换取认可”），需关联童年经历与当前行为

四、成年时期（当前年龄及以后：人格整合与发展期）
1. 当前生活状态与人格映射
- 分析当前职业/婚恋/学业状态如何体现其人格特质（如“无工作依赖家人→低尽责性的现实表现”“社交圈狭窄→低外向性的延续”）
- 核心心理困扰的显性表现（如“影响学习效率”“亲密关系频繁争吵”）
2. 人格发展趋势（分两种可能性）
- 自然发展趋势：不干预情况下，人格特质与心理困扰的演变（如“神经质偏高→情绪问题加剧”）
- 潜在成长空间：基于其人格优势（如高开放性、高宜人性）推导可突破的方向
3. 核心需求与未完成事件（精神分析落脚点）
- 明确当前未被满足的核心心理需求（如“被认可”“情感联结”“自主掌控感”）
- 指出童年/青年时期的“未完成事件”（如“未被父母看见的情绪”“亲密关系中的遗憾”），解释其对当前的影响

五、精神分析深度解读（强制4个维度，逻辑闭环）
1. 核心冲突本质（根源定位）
- 提炼人格内部/外部的核心冲突（如“依赖与独立的冲突”“对家庭的渴望与怨恨的冲突”），追溯至童年根源
2. 防御机制的适应性与局限性
- 分析防御机制在童年的“适应性作用”（如“回避→保护自己免受伤害”）
- 指出当前防御机制的“局限性”（如“过度回避→社交隔离”“讨好→自我需求压抑”）
3. 人格特质-经历-困扰的因果链（可视化逻辑）
- 用“童年经历→人格特质→青年事件→当前困扰”的链条总结，确保每个环节与原文强关联
4. 心理咨询切入点（实用导向）
- 基于阻抗程度、人格优势，提出1-2个具体咨询切入点（如“先通过共情建立信任，再探索潜意识信念”“利用高开放性引导自我觉察”）

格式强制要求（优先级最高，必须严格遵守）
1. 标题层级：仅保留纯文字标题（如“一、核心档案”“1. 早期客体关系”），禁止添加任何符号（包括#、*、**等）
2. 列表规范：核心档案、各模块下的子维度，仅允许用“-”引导列表，禁止用任何符号（*、+等）替代
3. 符号禁令：生成内容中**不得出现任何Markdown格式符号**，包括但不限于：*、**、#、[]、()（仅允许中文文字、数字、基本标点：逗号、句号、冒号、分号、顿号）
4. 内容约束：所有细节补充必须基于原文信息推导，不得脱离设定虚构（如原文无重大家庭矛盾，不得编造“父母离异”）
5. 逻辑要求：每个模块必须与前后模块形成因果呼应（如童年核心信念→青年冲突→当前困扰）
6. 语言风格：专业、客观，以精神分析视角切入，避免口语化，不使用情绪化表述

用户核心信息：
{user_profile}
"""

# LLM调用参数
LLM_PARAMS = {
    "model": openai_model,
    "temperature": 0.7,
    "max_tokens": 3000,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# -------------------------- 工具函数 --------------------------
def load_user_profile(input_path: Path) -> str:
    """读取TXT格式用户画像文件"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        # 过滤空行，保留原始信息结构
        content = "\n".join([line.strip() for line in f.readlines() if line.strip()])
    return content

def generate_biography(user_profile: str) -> str:
    """调用LLM生成精神分析人物小传（TXT格式内容）"""
    prompt = PROMPT_TEMPLATE.format(user_profile=user_profile)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **LLM_PARAMS
    )
    return response.choices[0].message.content.strip()

def save_biography(content: str, output_path: Path):
    """保存生成的小传为TXT文件"""
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

# -------------------------- 主逻辑（纯TXT批量处理）--------------------------
if __name__ == "__main__":
    print(f"开始批量生成{len(target_indices)}个人物小传（纯TXT输入输出）")
    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}\n")
    
    success_count = 0
    failed_tasks = []

    # 遍历目标文件索引
    for idx in target_indices:
        input_path = get_input_path(idx)
        output_path = get_output_path(idx)
        print(f"处理第 {idx} 个文件：")
        print(f"输入：{input_path.name}")
        print(f"输出：{output_path.name}")

        try:
            # 1. 读取TXT用户画像
            user_profile = load_user_profile(input_path)
            print(f"已读取用户画像（{len(user_profile)}字符）")

            # 2. 调用LLM生成小传
            print(f"正在生成人物小传...")
            biography = generate_biography(user_profile)

            # 3. 保存为TXT文件
            save_biography(biography, output_path)
            print(f"生成成功（约{len(biography)//2}字）\n")
            success_count += 1

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"处理失败：{error_msg}\n")
            failed_tasks.append((input_path.name, error_msg))

    # 输出处理总结
    print(f"批量处理完成！")
    print(f"成功生成：{success_count}/{len(target_indices)} 个TXT文件")
    if failed_tasks:
        print(f"失败文件：")
        for file, err in failed_tasks:
            print(f"  - {file}：{err}")