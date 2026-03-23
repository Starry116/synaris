import os
import shutil
from pathlib import Path

def copy_main_exe_from_source_to_target(source_root: str, target_root: str, exe_name: str = "main.exe"):
    """
    将源目录中每个项目文件夹内的 main.exe 复制到目标目录的同名项目文件夹中。
    """
    source_path = Path(source_root)
    target_path = Path(target_root)

    if not source_path.exists():
        print(f"错误：源目录不存在 - {source_path}")
        return
    if not target_path.exists():
        print(f"错误：目标目录不存在 - {target_path}")
        return

    count = 0
    # 遍历源目录下的所有子文件夹（项目）
    for project_folder in source_path.iterdir():
        if not project_folder.is_dir():
            continue

        main_exe = project_folder / exe_name
        if not main_exe.is_file():
            print(f"跳过：未找到 main.exe - {project_folder}")
            continue

        # 目标同名文件夹
        target_project = target_path / project_folder.name
        if not target_project.exists():
            print(f"警告：目标中不存在对应文件夹，已跳过 - {target_project}")
            continue

        target_exe = target_project / exe_name

        # 如果目标已存在 main.exe，可选择覆盖或跳过（此处默认覆盖，如需跳过请修改）
        try:
            shutil.copy2(str(main_exe), str(target_exe))
            print(f"成功复制：{main_exe} → {target_exe}")
            count += 1
        except Exception as e:
            print(f"错误：复制失败 {main_exe} → {target_exe}，原因：{e}")

    print(f"\n操作完成！共成功复制 {count} 个 main.exe 文件。")

# ====================== 使用配置 ======================
if __name__ == "__main__":
    SOURCE_DIR = r"\\192.168.0.200\share\游戏模拟工具\JILI逆向"
    TARGET_DIR = r"C:\Users\dp-pc\Desktop\testwork"

    copy_main_exe_from_source_to_target(SOURCE_DIR, TARGET_DIR)