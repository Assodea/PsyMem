import os

# 基础路径（请确保路径正确，使用原始字符串避免转义）
base_dir = r""

# 创建编号 101 到 150 的文件夹
for i in range(1, 31):
    folder_name = f"Consultation{i}"
    folder_path = os.path.join(base_dir, folder_name)
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ 已确保存在：{folder_path}")
    except Exception as e:
        print(f"✗ 创建失败 {folder_path}：{e}")