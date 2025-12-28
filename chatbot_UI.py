#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chatbot UI - 将 chatbot_tutorial.py 的命令行对话迁移到 Web UI
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import os
import re
import unicodedata

# ==================== 配置 ====================
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {device}")

# ==================== 常量 ====================
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10

# ==================== Voc类 ====================
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

# ==================== Encoder ====================
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

# ==================== Attention ====================
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

# ==================== Decoder ====================
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

# ==================== Greedy Decoder ====================
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

# ==================== 工具函数 ====================
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

# ==================== 加载模型 ====================
def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc_dict = checkpoint['voc_dict']

    voc = Voc('cornell movie-dialogs corpus')
    voc.__dict__ = voc_dict

    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    attn_model = 'dot'

    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)
    print("✅ Model loaded successfully!")
    return encoder, decoder, searcher, voc

# ==================== Web UI 对话函数 ====================
def chat_with_bot(encoder, decoder, searcher, voc, message, history):
    """Web UI 对话函数 - 将 chatbot_tutorial.py 的对话功能迁移到 Web"""
    try:
        if not message or not message.strip():
            return "Please say something!"

        # 句子归一化
        input_sentence = normalizeString(message)

        if not input_sentence:
            return "Please enter valid English text!"

        # 检查未知词
        words = input_sentence.split(' ')
        for word in words:
            if word not in voc.word2index:
                return f"Error: I don't know the word '{word}'"

        # 生成响应
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

        # 处理输出 - 去掉 EOS 和 PAD
        response_words = []
        for word in output_words:
            if word == 'EOS':
                break
            elif word != 'PAD':
                response_words.append(word)

        response = ' '.join(response_words) if response_words else "I don't understand."
        return response

    except KeyError as e:
        return f"Error: Encountered unknown word - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("  🤖 Chatbot UI - 启动中...")
    print("="*60)

    # 模型路径 - 请根据您的实际情况修改
    checkpoint_path = "data/save/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.tar"

    # 检查模型文件
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 找不到模型文件: {checkpoint_path}")
        print("\n请修改 checkpoint_path 为您的实际模型路径。")

        # 尝试自动查找模型文件
        print("\n正在查找模型文件...")
        import glob
        models = glob.glob("data/save/**/*.tar", recursive=True)
        if models:
            print(f"\n找到 {len(models)} 个模型文件:")
            for i, model in enumerate(models[:5], 1):
                print(f"  {i}. {model}")
            if len(models) > 5:
                print(f"  ... 还有 {len(models)-5} 个")
        return

    # 加载模型
    print(f"\n📦 正在加载模型...")
    encoder, decoder, searcher, voc = load_model(checkpoint_path)

    # 创建 Web UI 对话函数
    def chat_fn(message, history):
        return chat_with_bot(encoder, decoder, searcher, voc, message, history)

    # 创建Gradio界面 - 修复版：移除不兼容的参数
    print("\n🎨 创建Web界面...")
    try:
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="🤖 Chatbot UI - Ubuntu对话迁移到Web",
            description="""
            **将 chatbot_tutorial.py 的命令行对话迁移到 Web UI**

            💡 使用提示:
            - 请用**英文**输入
            - 句子要**简短** (少于10个词)
            - 使用**简单常见**的词汇

            📝 示例问题:
            • hello
            • how are you?
            • what is your name?
            • where are you from?
            • goodbye
            """,
            examples=[
                ["hello"],
                ["how are you?"],
                ["what is your name?"],
                ["where are you from?"],
                ["goodbye"]
            ],
        )
    except TypeError:
        # 兼容旧版 Gradio
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="🤖 Chatbot UI - Ubuntu对话迁移到Web",
        )

    # 启动服务
    print("\n" + "="*60)
    print("  🚀 Chatbot UI 已启动!")
    print("="*60)
    print(f"\n  📍 本机访问: http://localhost:7860")
    print(f"  🌐 局域网访问: http://你的IP:7860")
    print(f"\n  💡 按 Ctrl+C 停止服务")
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # DSW环境无法下载frpc，暂时关闭公网分享
        show_error=True,
        inbrowser=True  # 自动在浏览器打开
    )

if __name__ == "__main__":
    main()
