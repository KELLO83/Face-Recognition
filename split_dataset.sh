#!/bin/bash


SOURCE_DIR="/home/ubuntu/arcface-pytorch/pair_test"
DEST_DIR="/home/ubuntu/arcface-pytorch/pair"

mkdir -p "$DEST_DIR"


FOLDERS=($(find "$SOURCE_DIR" -maxdepth 1 -type d -not -path "$SOURCE_DIR"))

TOTAL_FOLDERS=${#FOLDERS[@]}
if [ $TOTAL_FOLDERS -eq 0 ]; then
    echo "Error: $SOURCE_DIR에 폴더가 없습니다."
    exit 1
fi


SELECT_COUNT=$(echo "$TOTAL_FOLDERS * 0.3" | bc | awk '{print int($1+0.5)}')

if [ $SELECT_COUNT -eq 0 ] && [ $TOTAL_FOLDERS -gt 0 ]; then
    SELECT_COUNT=1
fi

echo "총 폴더 수: $TOTAL_FOLDERS"
echo "선택할 폴더 수 : $SELECT_COUNT"

SELECTED_FOLDERS=($(printf "%s\n" "${FOLDERS[@]}" | shuf | head -n $SELECT_COUNT))

for FOLDER in "${SELECTED_FOLDERS[@]}"; do
    FOLDER_NAME=$(basename "$FOLDER")
    echo "이동 중: $FOLDER_NAME -> $DEST_DIR"
    mv "$FOLDER" "$DEST_DIR/"
    if [ $? -eq 0 ]; then
        echo "성공적으로 이동: $FOLDER_NAME"
    else
        echo "이동 실패: $FOLDER_NAME"
    fi
done

echo "작업 완료: $SELECT_COUNT 개 폴더가 $DEST_DIR로 이동되었습니다."