# ShadowDiffusion 依赖审计报告

生成日期: 2025-12-28

## 执行摘要

对项目的128个依赖进行了全面审计,发现:
- **5个包存在严重安全漏洞** (需要立即修复)
- **5个包使用本地文件路径** (导致可移植性问题)
- **约40个包完全未使用** (占总依赖的31%)
- **多个核心包严重过期** (2-3年未更新)

**建议操作**: 减少依赖数量从128个到约40个,并修复所有安全漏洞。

---

## 🔴 严重安全漏洞 (立即修复)

### 1. urllib3 1.26.12 → 2.6.0+
**影响**: 3个2025年CVE漏洞

- **CVE-2025-66471** (高危): 高度压缩数据的不当处理
  - 影响: 可能导致CPU耗尽和内存耗尽
  - 严重性: 高

- **CVE-2025-50181**: 重定向绕过导致SSRF
  - 影响: 当应用依赖禁用重定向来防止SSRF时,此漏洞会绕过保护

- **CVE-2025-66418**: 解压缩链无限制
  - 影响: DoS攻击向量

**修复**: `urllib3>=2.6.0`

**来源**:
- [NVD - CVE-2025-66471](https://nvd.nist.gov/vuln/detail/CVE-2025-66471)
- [CVE-2025-50181: urllib3 Redirect Bypass SSRF](https://www.miggo.io/vulnerability-database/cve/CVE-2025-50181)
- [urllib3 unbounded decompression chain](https://github.com/advisories/GHSA-gm62-xv2j-4w53)

### 2. aiohttp 3.8.5 → 3.12.14+
**影响**: CVE-2025-53643 - HTTP请求走私漏洞

- **CVE-2025-53643**: HTTP请求/响应走私
  - 影响: 可能绕过防火墙或代理保护
  - CVSS v4.0 分数: 1.7 (低)
  - 特别影响纯Python安装或启用AIOHTTPNOEXTENSIONS时

**修复**: `aiohttp>=3.12.14`

**注意**: 代码分析显示项目**未使用aiohttp**,建议完全移除此依赖。

**来源**:
- [CVE-2025-53643: AIOHTTP HTTP Request/Response Smuggling](https://vulert.com/vuln-db/pypi-aiohttp-195491)
- [aiohttp vulnerabilities | Snyk](https://security.snyk.io/package/pip/aiohttp)

### 3. requests 2.28.1 → 2.32.5+
**影响**: 2个CVE漏洞

- **CVE-2023-32681**: Proxy-Authorization头信息泄漏
  - 影响: 通过代理隧道连接时,认证凭据可能泄漏到目标服务器

- **CVE-2024-35195**: 证书验证绕过
  - 影响: 当第一个请求设置verify=False时,同一Session的后续请求会继续忽略证书验证

**修复**: `requests>=2.32.5`

**来源**:
- [requests 2.28.1 vulnerabilities | Snyk](https://security.snyk.io/package/pip/requests/2.28.1)
- [requests vulnerabilities | Snyk](https://security.snyk.io/package/pip/requests)

### 4. Pillow 9.2.0 → 10.0.0+
**影响**: CVE-2022-45199 - 拒绝服务

- **CVE-2022-45199**: 通过SAMPLESPERPIXEL导致DoS
  - 影响: 可以使应用崩溃
  - 修复版本: 9.3.0+

**修复**: `Pillow>=10.0.0` (建议使用最新稳定版)

**来源**:
- [pillow 9.2.0 vulnerabilities | Snyk](https://security.snyk.io/package/pip/pillow/9.2.0)
- [Python Pillow security vulnerabilities](https://www.cvedetails.com/product/27460/Python-Pillow.html)

### 5. Jinja2 3.1.2 → 3.1.6+
**影响**: CVE-2024-22195 - XSS漏洞

- **CVE-2024-22195**: 跨站脚本攻击 (XSS)
  - 影响: Jinja2拥有超过3300万周下载量,影响范围广
  - 修复版本: 3.1.3+
  - 最新版本: 3.1.6

**修复**: `Jinja2>=3.1.6`

**来源**:
- [Understanding and mitigating the Jinja2 XSS vulnerability](https://snyk.io/blog/jinja2-xss-vulnerability/)
- [Jinja2 3.1.2 vulnerabilities | Snyk](https://security.snyk.io/package/pip/Jinja2/3.1.2)

---

## ⚠️ 关键配置问题

### 问题1: 本地文件路径依赖
以下5个包使用本地conda构建路径,在其他环境中无法安装:

```python
certifi @ file:///opt/conda/conda-bld/certifi_1655968806487/work/certifi
mkl-random @ file:///tmp/build/80754af9/mkl_random_1626179032232/work
numpy @ file:///opt/conda/conda-bld/numpy_and_numpy_base_1653915516269/work
typing_extensions @ file:///tmp/abs_ben9emwtky/croots/recipe/typing_extensions_1659638822008/work
six @ file:///tmp/build/80754af9/six_1644875935023/work
```

**影响**:
- 无法在新环境中安装依赖
- CI/CD管道会失败
- 协作者无法复现环境

**解决方案**:
```python
certifi>=2024.0.0
numpy>=1.24.0
typing-extensions>=4.5.0
six>=1.16.0
# 移除mkl-random或使用可用版本
```

### 问题2: 不可用的包版本
- `mkl-fft==1.3.1` - PyPI上不存在(最低版本1.3.8)

**解决方案**:
```python
mkl-fft>=1.3.8  # 如果需要的话
```

---

## 🗑️ 未使用的依赖 (建议移除)

通过静态代码分析,以下约40个包在整个项目中**完全未被导入或使用**:

### Web框架 (8个包 - 0%使用率)
```
fastapi==0.88.0
uvicorn==0.22.0
starlette==0.22.0
starsessions==1.3.0
python-multipart==0.0.6
h11==0.14.0
itsdangerous==2.1.2
Werkzeug==2.2.2
```
**分析**: 项目是一个扩散模型,不需要web服务器。

### 异步HTTP (7个包 - 0%使用率)
```
aiohttp==3.8.5
aiosignal==1.3.1
async-timeout==4.0.2
asynctest==0.13.0
websocket-client==1.6.1
websockets==11.0.3
yarl==1.9.2
```
**分析**: 项目未使用异步HTTP客户端。

### Web爬取 (2个包 - 0%使用率)
```
beautifulsoup4==4.12.2
soupsieve==2.4.1
```

### PyTorch Lightning (3个包 - 0%使用率)
```
lightning==1.9.5
lightning-cloud==0.5.37
lightning-utilities==0.9.0
```
**分析**: 项目使用原生PyTorch,不使用Lightning。

### 数据可视化 (2个包 - 可能未使用)
```
matplotlib==3.5.3
pandas==1.3.5
```
**分析**: 未在代码中发现matplotlib或pandas的导入。

### 其他未使用的包 (~20个)
```
dominate==2.7.0
croniter==1.3.15
arrow==1.2.3
click==8.1.3
deepdiff==6.3.1
python-editor==1.0.4
markdown==3.4.1
markdown-it-py==2.2.0
rich==13.4.2
docker-pycreds==0.4.0
inquirer==2.10.1
blessed==1.20.0
readchar==4.0.5
fonttools==4.38.0
cycler==0.11.0
kiwisolver==1.4.4
mdurl==0.1.2
ordered-set==4.1.0
promise==2.3
exceptiongroup==1.1.2
```

**潜在节省**:
- 减少安装时间
- 减少磁盘空间占用
- 减少依赖冲突风险
- 简化依赖管理

---

## 📅 过期包

### PyTorch生态系统 (严重过期)
当前版本来自2022年,已过时2-3年:

```
torch==1.13.1          → 最新: 2.9.1 (2025年11月)
torchvision==0.8.2     → 需要与torch匹配
torchaudio==0.7.0a0    → 需要与torch匹配
```

**升级好处**:
- PyTorch 2.x 提供显著性能提升 (通过torch.compile)
- 100%向后兼容
- 更好的CUDA支持
- 众多bug修复和新特性

**迁移建议**:
```python
torch>=2.0.0,<3.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

**来源**:
- [PyTorch 2.9 Release Blog](https://pytorch.org/blog/pytorch-2-9/)
- [PyTorch Releases](https://github.com/pytorch/pytorch/releases)

### 其他过期的核心包

```
opencv-python==4.6.0.66    → 最新: 4.10.0+
scikit-image==0.19.3       → 最新: 0.24.0+
scikit-learn==1.0.2        → 最新: 1.5.0+
tensorboard==2.10.0        → 最新: 2.17.0+
tensorboardX==2.6          → 最新: 2.6.2.2
wandb==0.13.2              → 最新: 0.18.0+
```

---

## 📋 推荐的迁移策略

### 阶段1: 安全修复 (立即执行)
优先级: **关键**

1. 修复本地文件路径依赖:
```bash
# 创建清理后的requirements.txt
grep -v "@ file://" requirements.txt > requirements_clean.txt
```

2. 更新存在安全漏洞的包:
```bash
pip install --upgrade \
  'urllib3>=2.6.0' \
  'requests>=2.32.5' \
  'Pillow>=10.0.0' \
  'Jinja2>=3.1.6'
```

3. 测试核心功能确保兼容性

### 阶段2: 移除未使用的依赖
优先级: **高**

1. 使用推荐的requirements文件 (requirements.recommended.txt)
2. 在干净环境中测试:
```bash
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.recommended.txt
# 运行测试套件
python sr.py --help  # 或其他关键入口点
```

### 阶段3: 升级PyTorch (可选,需要测试)
优先级: **中**

1. 在单独的环境中测试PyTorch 2.x
2. 验证模型加载和训练
3. 检查CUDA兼容性
4. 性能基准测试

### 阶段4: 更新其他过期包
优先级: **低**

逐步更新其他包,每次更新后进行测试。

---

## 📊 统计对比

| 指标 | 当前 | 推荐 | 改善 |
|------|------|------|------|
| 总依赖数 | 128 | ~40 | -69% |
| 安全漏洞 | 5 | 0 | -100% |
| 可移植性问题 | 5 | 0 | -100% |
| 未使用的包 | ~40 | 0 | -100% |
| 平均包年龄 | ~2-3年 | <1年 | 显著改善 |

---

## ✅ 下一步行动

1. **立即执行**:
   - [ ] 审查 `requirements.recommended.txt`
   - [ ] 在开发环境中测试推荐的依赖
   - [ ] 修复安全漏洞

2. **短期** (本周):
   - [ ] 移除未使用的依赖
   - [ ] 修复本地文件路径问题
   - [ ] 更新CI/CD配置

3. **中期** (本月):
   - [ ] 评估PyTorch 2.x升级
   - [ ] 更新文档
   - [ ] 建立依赖审计流程

4. **持续**:
   - [ ] 定期运行 `pip-audit` 检查漏洞
   - [ ] 使用 Dependabot 或 Renovate 自动化依赖更新
   - [ ] 在添加新依赖前评估必要性

---

## 📚 参考资源

- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [PyTorch Migration Guide](https://pytorch.org/blog/pytorch-2-0-migration-guide/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Snyk Vulnerability Database](https://security.snyk.io/)

---

**报告生成者**: Claude Code
**审计工具**: pip-audit, 静态代码分析, Web安全数据库
**审计日期**: 2025-12-28
