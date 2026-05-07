import os
import re
import shutil
from pathlib import Path

def extract_dialogue_number(filename: str) -> int | None:
    """
    从文件名中提取对话编号（第一个数字段）
    支持格式：
      - 19_4.json          -> 19
      - memory_14_1.json   -> 14
      - re14_1.json        -> 14
      - re16_2.json        -> 16
    """
    # 去掉扩展名
    name = filename.replace('.json', '')
    # 使用正则匹配第一个连续数字
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    return None

def main():
    source_dir = Path("D:\VScodeProject\inter\interactive-agents-main\longdialogue")
    target_base = Path("")
    if not source_dir.exists():
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 获取所有 .json 文件
    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        print(f"⚠️ 在 {source_dir} 中没有找到 .json 文件")
        return
    
    moved_count = 0
    skipped_count = 0
    
    for file_path in json_files:
        filename = file_path.name
        dialogue_num = extract_dialogue_number(filename)
        
        if dialogue_num is None:
            print(f"⚠️ 无法提取编号，跳过: {filename}")
            skipped_count += 1
            continue
        
        target_dir = target_base / f"Consultation{dialogue_num}"
        
        # 如果目标目录不存在，跳过（因为之前已创建1-50，理论上存在）/home/cyr/interoriginal/ablation_remove_txt/Consultation1
        if not target_dir.exists():
            print(f"⚠️ 目标目录不存在，跳过: {target_dir} (文件: {filename})")
            skipped_count += 1
            continue
        
        # 目标完整路径（保留原文件名）
        dest_path = target_dir / filename
        
        # 如果目标文件已存在，可选择覆盖或跳过（这里采用跳过并提示）
        if dest_path.exists():
            print(f"⚠️ 目标文件已存在，跳过: {dest_path}")
            skipped_count += 1
            continue
        
        # 移动文件
        try:
            shutil.move(str(file_path), str(dest_path))
            print(f"✅ 移动: {filename} -> {target_dir}/")
            moved_count += 1
        except Exception as e:
            print(f"❌ 移动失败 {filename}: {e}")
            skipped_count += 1
    
    print(f"\n📊 完成：成功移动 {moved_count} 个文件，跳过 {skipped_count} 个文件。")

if __name__ == "__main__":
    main()