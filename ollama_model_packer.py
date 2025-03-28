#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama 模型打包工具

这个脚本允许你将 Ollama 模型打包成压缩归档文件。
它参考了 Ollama 的存储结构，能够准确找到模型相关的所有文件。
兼容 Windows 和 Linux 系统。

ollama version is 0.5.11
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
import platform
from datetime import datetime


def get_ollama_model_dir():
    """根据操作系统获取 Ollama 模型目录。"""
    # 首先检查环境变量
    env_models = os.environ.get("OLLAMA_MODELS")
    if env_models:
        return env_models

    # 如果环境变量未设置，则根据操作系统确定默认路径
    system = platform.system()

    if system == "Windows":
        # Windows 通常在用户主目录存储 Ollama 模型
        return os.path.join(os.path.expanduser("~"), ".ollama", "models")
    elif system == "Linux":
        # 检查 Linux 默认路径
        system_dir = os.path.join("/usr/share/ollama/", ".ollama", "models")
        user_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models")

        # 首先尝试系统路径
        if os.path.exists(system_dir):
            return system_dir
        return user_dir
    else:  # macOS 或其他系统
        return os.path.join(os.path.expanduser("~"), ".ollama", "models")


def get_model_list():
    """获取可用的 Ollama 模型列表。"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        # 跳过标题行和空行
        model_lines = [line for line in result.stdout.split('\n')[1:] if line.strip()]
        return model_lines
    except subprocess.CalledProcessError as e:
        print(f"执行 'ollama list' 时出错: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"意外错误: {e}")
        sys.exit(1)


def parse_model_name(model_line):
    """从 'ollama list' 输出的行中提取模型名称。"""
    parts = model_line.split()
    if not parts:
        return None
    return parts[0]  # 第一列是 NAME


def clean_model_name(model_name):
    """清理模型名称以用作文件名。"""

    # 将冒号、斜杠替换为下划线
    return model_name.replace('/', '_').replace(':', '_')


def find_model_files(model_name, base_dir):
    """根据模型名称找到所有相关文件的路径。

    查找模型清单和相关的 blob 文件。
    """
    # 将模型名拆分为名称和版本
    if ":" in model_name:
        name, version = model_name.split(":", 1)
    else:
        name = model_name
        version = "latest"  # 默认版本

    file_paths = []

    # 确定模型清单文件路径
    if "/" in name:
        # 处理用户分享的模型，如 modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF:latest
        username, modelname = name.split("/", 1)
        manifest_path = os.path.join(
            base_dir,
            "manifests",
            username,
            modelname,
            version
        )
    else:
        # 处理库模型，如 llama2
        manifest_path = os.path.join(
            base_dir,
            "manifests",
            "registry.ollama.ai",
            "library",
            name,
            version
        )

    # 检查清单文件是否存在
    if not os.path.exists(manifest_path):
        print(f"未找到模型清单文件: {manifest_path}")
        # 尝试寻找匹配的清单, 比如 quentinz/bge-large-zh-v1.5:latest
        manifests_dir = os.path.join(base_dir, "manifests", "registry.ollama.ai")
        if os.path.exists(manifests_dir):
            print("尝试寻找匹配的清单文件...")
            for root, dirs, files in os.walk(manifests_dir):
                for file in files:
                    if name.lower() in file.lower() or name.lower() in root.lower():
                        possible_path = os.path.join(root, file)
                        print(f"找到可能的清单文件: {possible_path}")
                        manifest_path = possible_path
                        break

    # 如果找到清单文件，添加到文件列表
    if os.path.exists(manifest_path):
        file_paths.append(manifest_path)

        # 尝试读取清单文件并找到所有相关的 blob 文件
        try:
            with open(manifest_path, 'r') as f:
                model_data = json.load(f)

            # 获取层数据
            layers = model_data.get("layers", [])

            # 添加配置信息
            if "config" in model_data:
                layers.append(model_data["config"])

            # 处理每一层
            for layer in layers:
                if isinstance(layer, dict) and "digest" in layer:
                    digest = layer["digest"]
                    # 替换冒号为连字符，符合 Ollama 的存储格式
                    digest = digest.replace(":", "-")
                    blob_path = os.path.join(base_dir, "blobs", digest)

                    if os.path.exists(blob_path):
                        file_paths.append(blob_path)
                    else:
                        print(f"警告: 未找到 blob 文件: {blob_path}")

        except Exception as e:
            print(f"读取模型清单文件出错: {e}")
            # 如果读取清单文件失败，尝试备用方法
            print("尝试备用方法查找模型文件...")

            # 尝试在 blobs 目录中查找与模型 ID 相关的文件
            model_id = get_model_id(model_name)
            blobs_dir = os.path.join(base_dir, "blobs")

            if os.path.exists(blobs_dir) and model_id:
                for filename in os.listdir(blobs_dir):
                    if model_id.lower() in filename.lower():
                        blob_path = os.path.join(blobs_dir, filename)
                        file_paths.append(blob_path)
    else:
        print(f"警告: 无法找到模型清单文件或适当的备用文件")

    return file_paths


def get_model_id(model_name):
    """从 ollama list 输出中提取模型 ID。"""
    models = get_model_list()
    for model_line in models:
        parts = model_line.split()
        if len(parts) >= 2 and parts[0] == model_name:
            return parts[1]  # 第二列是 ID
    return None


def package_model(model_name, output_dir=None):
    """将 Ollama 模型打包成压缩归档文件。"""
    # 如果未指定 output_dir，默认为当前目录
    if not output_dir:
        output_dir = os.getcwd()

    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)

    # 清理模型名称以用作文件名
    clean_name = clean_model_name(model_name)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建归档文件名
    archive_name = f"{clean_name}_{timestamp}.tar.gz"
    archive_path = os.path.join(output_dir, archive_name)

    print(f"正在将模型 '{model_name}' 打包到 '{archive_path}'...")

    # 获取 Ollama 模型目录
    ollama_dir = get_ollama_model_dir()
    print(f"在以下位置寻找模型文件: {ollama_dir}")

    if not os.path.exists(ollama_dir):
        print(f"未找到 Ollama 模型目录: {ollama_dir}")
        return False

    # 查找模型文件
    model_files = find_model_files(model_name, ollama_dir)

    if not model_files:
        print(f"在 {ollama_dir} 中找不到模型 '{model_name}' 的文件")
        return False

    print(f"为模型 '{model_name}' 找到 {len(model_files)} 个文件")

    try:
        # 创建临时目录进行打包
        temp_dir = os.path.join(output_dir, f"temp_{clean_name}_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        # 将模型文件复制到临时目录，保持相对路径结构
        for file_path in model_files:
            # 计算相对路径
            rel_path = os.path.relpath(file_path, ollama_dir)
            print(f"正在复制: {rel_path}")

            # 在临时目录中创建目标路径
            dest_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # 复制文件
            shutil.copy2(file_path, dest_path)

        # 创建归档文件
        base_name = os.path.splitext(archive_path)[0]
        if base_name.endswith('.tar'):
            base_name = base_name[:-4]

        shutil.make_archive(
            base_name,
            'gztar',  # 格式为 tar.gz
            root_dir=temp_dir,
            base_dir='.'
        )

        # 清理临时目录
        shutil.rmtree(temp_dir)

        print(f"成功将模型打包到 {archive_path}")
        return True

    except Exception as e:
        print(f"打包模型时出错: {e}")
        # 如果临时目录存在，清理它
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def list_models_interactive():
    """显示可用模型并让用户选择一个。"""
    models = get_model_list()

    if not models:
        print("未找到模型。")
        return None

    print("\n可用模型:")
    for i, model_line in enumerate(models, 1):
        print(f"{i}. {model_line}")

    while True:
        try:
            choice = input("\n输入要打包的模型编号（或输入 'q' 退出）: ")
            if choice.lower() == 'q':
                return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                model_name = parse_model_name(models[choice_idx])
                return model_name
            else:
                print("选择无效。请重试。")
        except ValueError:
            print("请输入数字或 'q' 退出。")


def main():
    parser = argparse.ArgumentParser(description="将 Ollama 模型打包成压缩归档文件")
    parser.add_argument("-m", "--model", help="要打包的 Ollama 模型名称")
    parser.add_argument("-o", "--output-dir", help="保存打包模型的目录，默认为当前目录", default=None)
    parser.add_argument("-l", "--list", action="store_true", help="列出可用模型")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式选择模型")
    parser.add_argument("--show-model-dir", action="store_true", help="显示您系统上的 Ollama 模型目录")

    args = parser.parse_args()

    if args.show_model_dir:
        model_dir = get_ollama_model_dir()
        print(f"Ollama 模型目录: {model_dir}")
        if os.path.exists(model_dir):
            print("目录存在")
            print(f"包含 {len(os.listdir(model_dir))} 个文件/目录")
        else:
            print("目录不存在")
        return

    if args.list:
        models = get_model_list()
        print("\n可用模型:")
        for model_line in models:
            print(model_line)
        return

    model_name = args.model

    if args.interactive or not model_name:
        model_name = list_models_interactive()
        if not model_name:
            print("未选择模型。退出。")
            return

    package_model(model_name, args.output_dir)


if __name__ == "__main__":
    main()
