if [ -z "$1" ]
then
    echo "Usage: $0 <file_list_txt>"
    exit 1
fi

if ! [ -f "$1" ]
then
     echo "File doesn't exist"
     exit 1
fi

file_list_txt="$1"
copy_count=0  # 初始化计数变量
line_count=$(wc -l < "$file_list_txt")

while IFS= read -r file_path; do
    ((copy_count++))
    # 获取文件名和目录路径
    file_name=$(basename "$file_path")
    directory_path=$(dirname "$file_path")

    directory_path=$(echo "$directory_path" | sed 's/raw_repos/repos/')

    # 创建目标目录（如果不存在）
    mkdir -p "$directory_path"

    # 构建目标文件路径
    destination_path="$directory_path/$file_name"

    # # 复制文件
    cp "$file_path" "$destination_path"

    echo "($copy_count/$line_count) File copied: $file_path to $destination_path"
done < "$file_list_txt"

echo "Copying complete."

