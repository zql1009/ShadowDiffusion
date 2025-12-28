# Chatbot Web界面使用说明

## 🎯 功能说明

将您的PyTorch命令行chatbot转换成Web界面，可以在浏览器中进行对话。

## 📋 前置条件

### 1. 已有训练好的模型
确保您已经训练好chatbot模型，并保存了checkpoint文件（.tar格式）

### 2. 安装依赖

```bash
pip install gradio torch
```

## 🚀 快速启动

### 方法1: 直接运行（最简单）

1. **修改模型路径**

   编辑 `chatbot_web.py` 文件，找到这一行：
   ```python
   checkpoint_path = "data/save/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.tar"
   ```

   修改为您实际的模型路径。

2. **启动服务**
   ```bash
   python chatbot_web.py
   ```

3. **打开浏览器**

   看到类似输出：
   ```
   Running on local URL:  http://127.0.0.1:7860
   Running on public URL: http://0.0.0.0:7860
   ```

   在浏览器中打开: `http://localhost:7860`

### 方法2: 从Jupyter Notebook迁移

如果您当前是在Jupyter Notebook中运行chatbot：

1. **保存当前训练的模型**

   在notebook中执行：
   ```python
   # 保存模型
   torch.save({
       'iteration': iteration,
       'en': encoder.state_dict(),
       'de': decoder.state_dict(),
       'en_opt': encoder_optimizer.state_dict(),
       'de_opt': decoder_optimizer.state_dict(),
       'loss': loss,
       'voc_dict': voc.__dict__,
       'embedding': embedding.state_dict()
   }, 'my_chatbot_model.tar')
   ```

2. **使用Web界面加载**

   修改 `chatbot_web.py` 中的路径：
   ```python
   checkpoint_path = "my_chatbot_model.tar"
   ```

3. **运行**
   ```bash
   python chatbot_web.py
   ```

## 🌐 访问方式

### 本机访问
```
http://localhost:7860
```

### 局域网访问（其他设备访问）

1. 查看Ubuntu的IP地址：
   ```bash
   ifconfig
   # 或
   ip addr show
   ```

2. 在其他设备浏览器中访问：
   ```
   http://你的Ubuntu的IP:7860
   ```
   例如: `http://192.168.1.100:7860`

### 公网访问（临时）

修改 `chatbot_web.py` 中的launch参数：
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # 改为True
)
```

Gradio会生成一个临时的公网链接（72小时有效），类似：
```
https://xxxxxxxxxxxx.gradio.live
```

## ⚙️ 自定义配置

### 修改端口
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=8080,  # 改为您想要的端口
)
```

### 修改界面主题
```python
demo = gr.ChatInterface(
    fn=chatbot.chat,
    theme="default",  # 可选: default, soft, glass, monochrome
    ...
)
```

### 添加密码保护
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("用户名", "密码")  # 添加登录验证
)
```

## 🐛 常见问题

### 1. 找不到模型文件
```
错误: 找不到模型文件 xxx.tar
```

**解决**: 检查checkpoint_path路径是否正确，使用绝对路径更保险。

### 2. 端口被占用
```
OSError: [Errno 98] Address already in use
```

**解决**: 修改端口号或关闭占用7860端口的程序：
```bash
# 查找占用端口的进程
lsof -i :7860
# 杀掉进程
kill -9 <PID>
```

### 3. CUDA out of memory

**解决**: 在代码开头强制使用CPU：
```python
USE_CUDA = False  # 改为False
device = torch.device("cpu")
```

### 4. 词典错误（Unknown word）

**解决**:
- 检查输入是否包含训练数据中没有的词
- 可以在normalizeString函数中处理未知词

## 📊 界面功能

### 1. 聊天框
- 输入消息后按Enter发送
- 支持多轮对话
- 保留对话历史

### 2. 示例问题
点击示例可以快速测试：
- "hello"
- "how are you?"
- "what is your name?"
等

### 3. 控制按钮
- **删除上一条**: 撤销最后一条对话
- **清空对话**: 清除所有历史记录
- **提交**: 发送消息

## 🔧 进阶: 部署为系统服务

如果想让chatbot在后台持续运行：

### 使用systemd（推荐）

1. 创建服务文件：
```bash
sudo nano /etc/systemd/system/chatbot.service
```

2. 添加内容：
```ini
[Unit]
Description=Chatbot Web Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/your/chatbot
ExecStart=/usr/bin/python3 /path/to/your/chatbot/chatbot_web.py
Restart=always

[Install]
WantedBy=multi-user.target
```

3. 启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable chatbot
sudo systemctl start chatbot
```

### 使用screen（简单）

```bash
# 创建新session
screen -S chatbot

# 运行程序
python chatbot_web.py

# 按 Ctrl+A 然后按 D 离开screen
# 程序会在后台继续运行

# 回到session
screen -r chatbot
```

## 📝 完整示例

```python
# 最小化示例
import gradio as gr

def chat(message, history):
    # 您的chatbot逻辑
    response = your_chatbot_function(message)
    return response

demo = gr.ChatInterface(fn=chat)
demo.launch(server_name="0.0.0.0", server_port=7860)
```

## 🎨 界面截图

启动后会看到类似这样的界面：
```
┌─────────────────────────────────────┐
│  🤖 PyTorch Chatbot                 │
│  基于Cornell Movie Dialogs训练      │
├─────────────────────────────────────┤
│  [示例问题]                          │
│  • hello                            │
│  • how are you?                     │
│  • what is your name?               │
├─────────────────────────────────────┤
│  User: hello                        │
│  Bot:  hello .                      │
│                                     │
│  User: how are you?                 │
│  Bot:  i m fine .                   │
├─────────────────────────────────────┤
│  [输入框]                            │
│  [删除上一条] [清空对话] [提交]      │
└─────────────────────────────────────┘
```

## 💡 提示

1. **第一次启动会比较慢**，因为要加载模型
2. **使用GPU会更快**，但不是必须的
3. **对话质量取决于训练数据**，可以用更多数据重新训练
4. **建议使用Chrome或Firefox浏览器**访问

## 📞 需要帮助？

如果遇到问题：
1. 检查Python版本 >= 3.7
2. 检查PyTorch是否正确安装
3. 检查模型文件路径
4. 查看终端的错误信息
