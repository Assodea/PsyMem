import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
#python Turing.py
class PsychTuringTestAnalyzer:
    def __init__(self, base_path: str, group_id: int, verbose: bool = False):
        """verbose=False 时不在控制台打印详细分析过程"""
        self.client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )
        self.model = 'gpt-4o'
        self.base_path = base_path
        self.group_id = group_id
        self.group_folder = os.path.join(base_path, f"Consultation{group_id}")
        self.sessions = []
        self.result = None
        self.error_message = None
        self.verbose = verbose

    def find_memory_files(self) -> List[str]:
        files = []
        for i in range(1, 5):
            filename = f"memory_{self.group_id}_{i}.json"
            file_path = os.path.join(self.group_folder, filename)
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                self.error_message = f"文件缺失: {filename}"
                return []
        return files

    def load_sessions(self, file_paths: List[str]):
        self.sessions = []
        for i, file_path in enumerate(file_paths, 1):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    session_data['session_number'] = i
                    session_data['file_name'] = os.path.basename(file_path)
                    self.sessions.append(session_data)
            except Exception as e:
                self.error_message = f"读取文件失败 {file_path}: {e}"
                raise

    def prepare_prompt(self) -> str:
        sessions_text = ""
        for session in self.sessions:
            session_num = session['session_number']
            client_memory = session.get('client_memory', {})
            client_facts = "\n".join([f"    - {fact}" for fact in client_memory.get('facts', [])])
            client_issues = "\n".join([f"    - {issue}" for issue in client_memory.get('issues', [])])
            client_personality = client_memory.get('personality', '无记录')
            counselor_memory = session.get('counselor_memory', {})
            counselor_facts = "\n".join([f"    - {fact}" for fact in counselor_memory.get('facts', [])])

            sessions_text += f"""
第{session_num}次心理咨询记录（{session.get('timestamp', '无时间戳')}）：
来访者记忆：
1. 事实记忆：
{client_facts if client_facts else "    - 无记录"}
2. 关键议题：
{client_issues if client_issues else "    - 无记录"}
3. 对咨询师的印象：
{client_personality}

咨询师观察记录：
{counselor_facts if counselor_facts else "    - 无记录"}

"""
        prompt = f"""请分析以下4次心理咨询的记忆记录，判断这些记忆是否属于同一个来访者。

输入数据：
{sessions_text}

请仅输出JSON格式的结果，结构如下：
{{
  "identity_consistency": {{
    "consistent": true/false,
    "confidence": 0-100,
    "evidence": ["证据1", "证据2"],
    "contradictions": ["矛盾点1"],
    "summary": "综合判断说明"
  }}
}}

要求：
- consistent为true表示4次记忆属于同一个人，false表示不是同一个人。
- confidence为置信度（0-100）,80分以上可判断为同一人
- evidence列出支持判断的关键证据。
- contradictions列出任何矛盾或不一致之处。
- summary用一句话总结结论。

请确保使用中文输出。"""
        return prompt

    def call_gpt4o(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的心理咨询分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.error_message = f"API调用失败: {e}"
            return None

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                try:
                    return json.loads(response_text[start:end])
                except:
                    pass
        self.error_message = "无法从GPT响应中解析JSON"
        return {}

    def run_analysis(self) -> Optional[Dict]:
        """返回分析结果字典（包含consistent, confidence, summary），失败返回None"""
        if self.verbose:
            print(f"\n开始分析 Consultation{self.group_id} 的四次记忆...")
        os.makedirs(self.group_folder, exist_ok=True)

        file_paths = self.find_memory_files()
        if not file_paths:
            self.save_error_report()
            return None

        try:
            self.load_sessions(file_paths)
        except Exception:
            self.save_error_report()
            return None

        if self.verbose:
            print(f"已加载 {len(self.sessions)} 个咨询会话")
        prompt = self.prepare_prompt()
        response_text = self.call_gpt4o(prompt)
        if response_text is None:
            self.save_error_report()
            return None

        self.result = self.extract_json_from_response(response_text)
        if not self.result:
            self.save_error_report()
            return None

        # 保存详细结果文件
        self.save_results()
        if self.verbose:
            self.display_summary()

        identity = self.result.get("identity_consistency", {})
        return {
            "consistent": identity.get("consistent"),
            "confidence": identity.get("confidence", 0.0),
            "summary": identity.get("summary", "")
        }

    def save_error_report(self):
        txt_file = os.path.join(self.group_folder, "identity_report.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"心理咨询记忆身份一致性分析报告 (Consultation{self.group_id})\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("分析失败\n")
            f.write(f"失败原因: {self.error_message or '未知错误'}\n")
            f.write("\n请检查以下事项：\n")
            f.write("1. 确认该文件夹下存在 memory_X_1.json ~ memory_X_4.json 四个文件\n")
            f.write("2. 确认文件内容格式正确（可被JSON解析）\n")
            f.write("3. 确认Azure OpenAI环境变量配置正确，网络通畅\n")
            f.write("=" * 60 + "\n")
        if self.verbose:
            print(f"错误报告已保存: {txt_file}")

    def save_results(self):
        json_file = os.path.join(self.group_folder, "identity_test.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"JSON结果已保存: {json_file}")
        self.save_txt_report()

    def save_txt_report(self):
        txt_file = os.path.join(self.group_folder, "identity_report.txt")
        identity = self.result.get("identity_consistency", {})
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"心理咨询记忆身份一致性分析报告 (Consultation{self.group_id})\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("判定结果: " + ("属于同一个人" if identity.get('consistent') else "不属于同一个人") + "\n")
            f.write(f"置信度: {identity.get('confidence', 0):.3f}\n\n")
            if identity.get("evidence"):
                f.write("支持证据:\n")
                for ev in identity.get("evidence", []):
                    f.write(f"- {ev}\n")
                f.write("\n")
            if identity.get("contradictions"):
                f.write("矛盾点:\n")
                for cont in identity.get("contradictions", []):
                    f.write(f"- {cont}\n")
                f.write("\n")
            if identity.get("summary"):
                f.write(f"综合结论: {identity.get('summary')}\n")
            f.write("\n" + "=" * 60 + "\n")
        if self.verbose:
            print(f"TXT报告已保存: {txt_file}")

    def display_summary(self):
        identity = self.result.get("identity_consistency", {})
        print(f"\n结果 (Consultation{self.group_id}): {'一致' if identity.get('consistent') else '不一致'} (置信度: {identity.get('confidence', 0):.3f})")
        print(f"结论: {identity.get('summary', '无')}")

def main():
    print("-" * 60)
    print("心理咨询记忆身份一致性分析（图灵测试）")
    print("-" * 60)
    base_path = r"D:\VScodeProject\data\2muti_dialogue2"
#D:\VScodeProject\data\ablation_remove_mem
#D:\VScodeProject\data\oursOutput150
#D:\VScodeProject\data\2muti_dialogue2

##python Turing.py
    if not all([os.environ.get("AZURE_OPENAI_ENDPOINT"), 
                os.environ.get("AZURE_OPENAI_API_KEY"), 
                os.environ.get("AZURE_OPENAI_API_VERSION")]):
        print("错误: 请设置Azure OpenAI环境变量")
        return

    start_group, end_group = 1, 50
    user_input = input(f"请输入要分析的组范围 (默认 1-50)：").strip()
    if user_input:
        try:
            if '-' in user_input:
                parts = user_input.split('-')
                start_group = int(parts[0].strip())
                end_group = int(parts[1].strip())
            else:
                start_group = end_group = int(user_input)
        except:
            print("输入格式错误，使用默认范围 1-50")

    results = []   # 存储成功组的结果 {"group": int, "consistent": bool, "confidence": float, "summary": str}
    failed_groups = []  # 存储失败的组号

    for group_id in range(start_group, end_group + 1):
        # verbose=False 关闭中间打印
        analyzer = PsychTuringTestAnalyzer(base_path, group_id, verbose=False)
        res = analyzer.run_analysis()
        if res is not None:
            results.append({
                "group": group_id,
                "consistent": res["consistent"],
                "confidence": res["confidence"],
                "summary": res["summary"]
            })
        else:
            failed_groups.append(group_id)
        # 可选：打印进度点
        print(f"已完成 {group_id}/{end_group}", end='\r')

    print("\n\n" + "=" * 80)
    print("最终汇总报告")
    print("=" * 80)

    # 打印成功组的结果表格
    if results:
        print("\n成功分析组数: {} / {}".format(len(results), end_group - start_group + 1))
        print("\n{:<5} {:<10} {:<10} {}".format("组号", "一致?", "置信度", "结论摘要"))
        print("-" * 80)
        for r in results:
            consistent_str = "是" if r["consistent"] else "否"
            print("{:<5} {:<10} {:<10.3f} {}".format(r["group"], consistent_str, r["confidence"], r["summary"][:50] + "..." if len(r["summary"]) > 50 else r["summary"]))
        
        # 计算平均置信度
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        print("\n" + "-" * 80)
        print(f"平均置信度: {avg_confidence:.4f}")
    else:
        print("没有成功分析的组")

    if failed_groups:
        print(f"\n失败组: {failed_groups} (共 {len(failed_groups)} 组)")
        print("失败原因请查看对应组文件夹内的 identity_report.txt")

    # 可选：保存汇总报告到文件
    summary_file = os.path.join(base_path, "identity_analysis_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"心理咨询记忆身份一致性分析汇总报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析范围: Consultation{start_group} ~ Consultation{end_group}\n")
        f.write(f"成功组数: {len(results)}/{end_group - start_group + 1}\n")
        if results:
            f.write(f"平均置信度: {avg_confidence:.4f}\n\n")
            f.write("详细结果:\n")
            f.write("{:<5} {:<10} {:<10} {}\n".format("组号", "一致?", "置信度", "结论摘要"))
            for r in results:
                consistent_str = "是" if r["consistent"] else "否"
                f.write("{:<5} {:<10} {:<10.3f} {}\n".format(r["group"], consistent_str, r["confidence"], r["summary"]))
        if failed_groups:
            f.write(f"\n失败组: {failed_groups}\n")
    print(f"\n汇总报告已保存至: {summary_file}")

if __name__ == "__main__":
    main()