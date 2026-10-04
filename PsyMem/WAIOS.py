"""
WAI-O-S 心理咨询对话批量评估脚本 (基于精简版修改)
使用Azure OpenAI GPT-4o模型评估指定目录下所有Consultation子文件夹内的JSON对话文件

分两个阶段：
1. 评估所有X_1.json（约50个）
2. 评估所有X_1.json ~ X_4.json（约200个）

结果保存为TXT摘要文件，包含每个文件的评分和维度平均值。
在评估前输出各阶段找到的文件数目和文件名列表。
"""

import os
import json
import re
from datetime import datetime
from openai import AzureOpenAI
#python oursWAIOS.py
# =============== 配置部分 ===============
# Azure OpenAI客户端配置
openai_client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)
openai_model = 'gpt-4o'

# 根目录（包含Consultation1, Consultation2, ...）
ROOT_DIR = r"D:\VScodeProject\data\ablation_remove_mem"
#D:\VScodeProject\data\ablation_remove_re
OUTPUT_DIR = "waios_results"

# =============== 核心评估函数（保持原样） ===============
def convert_dialogue_to_text(dialogue):
    """将对话列表转换为纯文本格式"""
    return "\n".join([
        f"{'来访者' if item.get('role') == 'client' else '咨询师'}: {item.get('content', '')}"
        for item in dialogue
    ])

def generate_evaluation_prompt(conversation_text):
    """生成WAI-O-S评估提示词"""
    return f"""请作为心理咨询评估专家，使用WAI-O量表评估以下对话：

{conversation_text}

请为15个项目评分(1-7分，1=非常强的反面证据，7=非常强的正面证据)：

Goal维度（目标一致）：
1. [2]双方在改善来访者状况的步骤上达成一致
2. [6]双方对治疗目标有共同认知
3. [10]双方在会谈目标上存在分歧[R]
4. [24]双方对来访者需要重点解决的问题达成一致
5. [27]双方对来访者的真实问题有不同看法[R]

Task维度（任务一致）：
6. [4]双方对当前治疗活动的有用性达成一致
7. [13]双方对来访者在治疗中的责任达成一致
8. [16]双方相信当前的治疗活动能帮助来访者实现想要的改变
9. [22]双方正在共同商定的目标上工作
10. [30]双方共同设定了会谈目标

Bond维度（关系联结）：
11. [1]关系中存在不适感[R]
12. [8]双方相互喜欢
13. [19]双方相互尊重
14. [26]双方相互信任
15. [36]来访者感到咨询师尊重并关心自己

请按顺序输出15个评分，每行一个数字：
1. [项目2评分]
2. [项目6评分]
... (15行)"""

def parse_scores(response_text):
    """从API响应中解析15个评分"""
    scores = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        match = re.search(r'(\d+)\.?\s*[:：]?\s*(\d+)', line)
        if match and 1 <= int(match.group(2)) <= 7:
            scores.append(int(match.group(2)))
            if len(scores) == 15:
                break
    

def calculate_waios_scores(ratings):
    """计算WAI-O-S三个维度的分数"""
    # 反转计分（反向项目：索引2,4,10对应项目10,27,1）
    adjusted = ratings.copy()
    for idx in [2, 4, 10]:
        adjusted[idx] = 8 - adjusted[idx]
    
    # 计算各维度平均分
    goal_avg = sum(adjusted[0:5]) / 5
    task_avg = sum(adjusted[5:10]) / 5
    bond_avg = sum(adjusted[10:15]) / 5
    overall_avg = (goal_avg + task_avg + bond_avg) / 3
    
    return {
        "goal": round(goal_avg, 2),
        "task": round(task_avg, 2),
        "bond": round(bond_avg, 2),
        "overall": round(overall_avg, 2),
        "raw_scores": ratings
    }

def get_quality_level(score):
    """根据分数返回质量等级"""
    if score >= 6.0: return "优秀"
    elif score >= 5.0: return "良好"
    elif score >= 4.0: return "中等"
    else: return "较差"

def load_dialogue_file(file_path):
    """加载单个JSON对话文件（支持列表或嵌套格式）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种JSON格式
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in ["session", "dialogues", "conversation", "dialogue"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
        
        return []
    except Exception as e:
        print(f"  加载失败: {e}")
        return []

def evaluate_dialogue(file_path, file_name):
    """评估单个对话文件"""
    print(f"  正在评估文件 {file_name}...")
    
    try:
        # 加载对话
        dialogue = load_dialogue_file(file_path)
        if not dialogue:
            return {"success": False, "file_name": file_name, "error": "文件为空或格式错误"}
        
        # 转换格式并生成提示词
        text = convert_dialogue_to_text(dialogue)
        prompt = generate_evaluation_prompt(text)
        
        # 调用Azure OpenAI API
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是专业的心理咨询评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        # 解析评分并计算分数
        ratings = parse_scores(response.choices[0].message.content)
        scores = calculate_waios_scores(ratings)
        
        return {
            "success": True,
            "file_name": file_name,
            "length": len(dialogue),
            "turns": text.count("来访者:"),
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  评估失败: {e}")
        return {"success": False, "file_name": file_name, "error": str(e)}

# =============== 新增：文件收集与批量处理 ===============
def collect_files_by_phase(phase):
    """
    根据阶段收集文件路径列表
    phase=1: 所有 X_1.json
    phase=2: 所有 X_1.json, X_2.json, X_3.json, X_4.json
    """
    files = []
    # 遍历根目录下所有以Consultation开头的文件夹
    for item in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, item)
        if not os.path.isdir(folder_path) or not item.startswith('Consultation'):
            continue
        match = re.match(r'Consultation(\d+)', item)
        if not match:
            continue
        folder_num = int(match.group(1))
        # 遍历文件夹内JSON文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path) or not filename.endswith('.json'):
                continue
            m = re.match(r'^(\d+)_([1-4])\.json$', filename)
            if not m:
                continue
            file_num = int(m.group(1))
            file_sub = int(m.group(2))
            # 确保文件编号与文件夹编号一致
            if file_num != folder_num:
                continue
            if phase == 1 and file_sub == 1:
                files.append((file_path, filename))
            elif phase == 2 and file_sub in [1,2,3,4]:
                files.append((file_path, filename))
    # 按文件夹数字和文件数字排序
    files.sort(key=lambda x: (
        int(re.search(r'Consultation(\d+)', x[0]).group(1)),
        int(re.search(r'_(\d+)\.json$', x[0]).group(1))
    ))
    return files

def batch_evaluate(file_list, phase_name):
    """批量评估一组文件，返回结果列表"""
    results = []
    total = len(file_list)
    for i, (file_path, file_name) in enumerate(file_list, 1):
        print(f"[{phase_name}] 处理第 {i}/{total} 个文件: {file_name}")
        result = evaluate_dialogue(file_path, file_name)
        results.append(result)
        if result["success"]:
            s = result["scores"]
            level = get_quality_level(s["overall"])
            print(f"  ✓ Goal={s['goal']:.2f}, Task={s['task']:.2f}, Bond={s['bond']:.2f}, Overall={s['overall']:.2f} ({level})")
        else:
            print(f"  ✗ 评估失败: {result.get('error', '未知错误')}")
    return results

def generate_summary_txt(results, filename, title):
    """生成包含详细评分和平均值的TXT摘要文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    successful = [r for r in results if r["success"]]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write("="*60 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {len(results)}\n")
        f.write(f"成功评估: {len(successful)}\n")
        f.write(f"失败: {len(results)-len(successful)}\n\n")
        if successful:
            f.write("详细评分:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'文件名':<30} {'Goal':<8} {'Task':<8} {'Bond':<8} {'Overall':<8} {'等级'}\n")
            for r in successful:
                s = r["scores"]
                level = get_quality_level(s["overall"])
                f.write(f"{r['file_name']:<30} {s['goal']:<8.2f} {s['task']:<8.2f} {s['bond']:<8.2f} {s['overall']:<8.2f} {level}\n")
            # 计算平均值
            avg_goal = sum(r["scores"]["goal"] for r in successful) / len(successful)
            avg_task = sum(r["scores"]["task"] for r in successful) / len(successful)
            avg_bond = sum(r["scores"]["bond"] for r in successful) / len(successful)
            avg_overall = sum(r["scores"]["overall"] for r in successful) / len(successful)
            f.write("-"*60 + "\n")
            f.write(f"{'平均值':<30} {avg_goal:<8.2f} {avg_task:<8.2f} {avg_bond:<8.2f} {avg_overall:<8.2f}\n")
        else:
            f.write("没有成功评估的文件。\n")
    print(f"摘要已保存: {filepath}")

# =============== 主程序 ===============
def main():
    print("="*60)
    print("WAI-O-S 心理咨询对话批量评估 - 新版目录结构")
    print("="*60)

    # 检查环境变量
    if not all(os.environ.get(var) for var in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]):
        print("错误: 请设置Azure OpenAI环境变量")
        return

    # 第一阶段：收集所有 X_1.json
    print("\n第一阶段：收集所有 X_1.json 文件...")
    files_phase1 = collect_files_by_phase(1)
    print(f"找到 {len(files_phase1)} 个文件（期望约50个），文件列表：")
    for _, fname in files_phase1:
        print(f"  {fname}")
    results1 = batch_evaluate(files_phase1, "第一阶段")

    # 第二阶段：收集所有 X_1.json ~ X_4.json
    print("\n第二阶段：收集所有 X_1~X_4.json 文件...")
    files_phase2 = collect_files_by_phase(2)
    print(f"找到 {len(files_phase2)} 个文件（期望约200个），文件列表：")
    for _, fname in files_phase2:
        print(f"  {fname}")
    results2 = batch_evaluate(files_phase2, "第二阶段")

    # 生成TXT摘要
    generate_summary_txt(results1, "phase1_summary.txt", "第一阶段（所有 X_1.json）评估结果")
    generate_summary_txt(results2, "phase2_summary.txt", "第二阶段（所有 X_1~X_4.json）评估结果")

    print(f"\n✓ 评估完成！摘要文件已保存至 '{OUTPUT_DIR}' 目录。")

if __name__ == "__main__":
    main()
