#!/bin/bash

# 参数：A 目录和 B 目录
A_DIR="/app/joern-cli/lib/"
B_DIR="/app/RepoSPD/joern/lib/"

# 检查参数是否正确
if [ -z "$A_DIR" ] || [ -z "$B_DIR" ]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

# 确保 A 和 B 是有效目录
if [ ! -d "$A_DIR" ]; then
  echo "Error: Source directory $A_DIR does not exist."
  exit 1
fi

if [ ! -d "$B_DIR" ]; then
  echo "Error: Destination directory $B_DIR does not exist."
  exit 1
fi

# 遍历 A 目录中的文件
find "$A_DIR" -type f | while read -r file; do
  # 获取文件名
  filename=$(basename "$file")
  destination="$B_DIR/$filename"

  # 调试输出文件路径
  echo "Checking file: $file"
  echo "Destination: $destination"

  # 判断 B 目录中是否已存在该文件，且大小不为零
  if [ -e "$destination" ] && [ -s "$destination" ]; then
    echo " "
  else
    # 复制文件
    cp "$file" "$B_DIR"
    echo "Copied $filename to $B_DIR."
  fi
done
