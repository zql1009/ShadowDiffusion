#!/bin/bash
# Chatbot Web启动脚本

echo "================================"
echo "   Chatbot Web 启动器"
echo "================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

echo "选择启动方式:"
echo "1) Gradio界面 (推荐，自动生成聊天UI)"
echo "2) Flask界面 (自定义HTML界面)"
echo ""
read -p "请选择 (1/2): " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动Gradio版本..."
        echo "访问地址: http://localhost:7860"
        echo ""
        python3 chatbot_web.py
        ;;
    2)
        echo ""
        echo "🚀 启动Flask版本..."
        echo "访问地址: http://localhost:5000"
        echo ""
        python3 chatbot_flask.py
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
