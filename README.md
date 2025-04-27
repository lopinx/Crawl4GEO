<p align="center">
  <a href="https://github.com/lopinx/crawl4geo" target="_blank"><img src="https://cdn.lightpanda.io/assets/images/logo/lpd-logo.png" alt="Logo" height=170></a>
</p>

<h1 align="center">GEO数据爬取助手</h1>

<p align="center"><a href="https://github.com/lopinx/crawl4geo">GEO数据爬取助手</a></p>

<div align="center">

[![KeyBERT](https://img.shields.io/github/stars/MaartenGr/KeyBERT)](https://github.com/MaartenGr/KeyBERT)
[![jieba](https://img.shields.io/github/stars/fxsjy/jieba)](https://github.com/fxsjy/jieba)
[![NLTK](https://img.shields.io/github/stars/nltk/nltk)](https://github.com/nltk/nltk)

</div>

---

# Crawl4GEO

## 项目概述

`Crawl4GEO` 是一个用于从多个站点爬取文章并生成结构化数据的自动化脚本。该脚本支持多站点爬取，使用 `KeyBERT` 和 `TextRank`, `jiba` 等算法进行关键词提取，并支持将爬取的文章导出为 Markdown 格式，方便集成到各种内容管理系统（CMS/LLM）中。

## 安装依赖

### 使用 `uv` 工具安装依赖

1. 安装 `uv` 工具（如果尚未安装）：
   ```bash
   pip install uv
   ```

2. 使用 `uv` 添加依赖包：
   ```bash
   uv add aiofiles aiosqlite "httpx[http2,http3]" keybert scikit-learn jieba nltk toml lxml bs4 markdown markdownify pillow python-slugify pypinyin
   ```

3. 或者使用 `requirements.txt` 安装依赖：
   ```bash
   uv pip install -r requirements.txt
   ```

4. 导出依赖包：
   ```bash
   uv pip freeze | uv pip compile - -o requirements.txt
   ```

### 手动安装依赖

你也可以手动安装所有依赖包：
```bash
pip install aiofiles aiosqlite "httpx[http2,http3]" keybert scikit-learn jieba nltk toml lxml bs4 markdown markdownify pillow python-slugify pypinyin
```

## 配置说明

### `config.json`

`config.json` 文件包含爬虫的配置信息，包括站点信息、选择器、过滤规则等。以下是一个示例配置：

```json
[
    {
        "start_urls": ["https://example.com/articles"],
        "user_agent": "lopins.cn/0.1 Fetcher, support@lopins.cn",
        "selectors": {
            "article": {
                "title": "h1.title",
                "content": "div.content",
                "excerpt": "meta[name='description']",
                "tags": "meta[name='keywords']",
                "date": "time.published-date",
                "extras": {
                    "author": "span.author"
                }
            },
            "list": {
                "article": "a.article-link",
                "next": "a.next-page"
            },
            "remove": {
                "tags": ["div.ads", "div.sidebar"],
                "regex": ["广告", "赞助"]
            },
            "filter": {
                "titles": ["广告", "赞助"],
                "words": ["广告", "赞助"]
            }
        },
        "tokenizer": {
            "require": ["关键词1", "关键词2"],
            "filter": ["过滤词1", "过滤词2"]
        },
        "watermark": {
            "text": "版权信息"
        },
        "slugify": {
            "separator": "-",
            "stopwords": ["的", "了", "是"],
            "replacements": [["&", "and"]]
        },
        "categories": ["技术", "新闻"],
        "author": "lopins",
        "cover": "https://example.com/cover.jpg",
        "cms": "hexo",
        "amount": 5
    }
]
```

## 使用说明

### 运行爬虫

1. 确保已安装所有依赖。

2. 配置 `config.json` 文件。

3. 运行脚本：

   ```bash
   python main.py
   ```

### 日志记录

日志文件将保存在每个站点的目录下，文件名为 `bot.log`。例如：

```
\example.com\bot.log
```

### 数据库

爬取的数据将保存在每个站点的 SQLite 数据库中，文件名为 `log.db`。例如：

```
\example.com\log.db
```

### 导出文件

生成的 Markdown 文件将保存在每个站点的 `article` 目录下。例如：

```
\example.com\article\20231001-12345678-12345678.md
```

## 代码结构

```
Crawl4GEO/
├── main.py
├── dev.json
├── pyproject.toml
├── requirements.txt
├── ChineseStopWords.txt
├── EnglishStopWords.txt
└── example.com/
    ├── articles/
        └── 20231001-12345678-12345678.md
    ├── uploads/
        └── md520231001-12345678-12345678.jpg
    ├── bot.log
    └── log.log
```

## 常见问题

### Q: 如何添加新的站点？

A: 编辑 `dev.json` 文件，添加新的站点配置。例如：

```json
{
    "start_urls": ["https://newsite.com/articles"],
    "user_agent": "lopins.cn/0.1 Fetcher, support@lopins.cn",
    "selectors": {
        "article": {
            "title": "h1.title",
            "content": "div.content",
            "excerpt": "meta[name='description']",
            "tags": "meta[name='keywords']",
            "date": "time.published-date",
            "extras": {
                "author": "span.author"
            }
        },
        "list": {
            "article": "a.article-link",
            "next": "a.next-page"
        },
        "remove": {
            "tags": ["div.ads", "div.sidebar"],
            "regex": ["广告", "赞助"]
        },
        "filter": {
            "titles": ["广告", "赞助"],
            "words": ["广告", "赞助"]
        }
    },
    "tokenizer": {
        "require": ["关键词1", "关键词2"],
        "filter": ["过滤词1", "过滤词2"]
    },
    "watermark": {
        "text": "版权信息"
    },
    "slugify": {
        "separator": "-",
        "stopwords": ["的", "了", "是"],
        "replacements": [["&", "and"]]
    },
    "categories": ["技术", "新闻"],
    "author": "lopins",
    "cover": "https://newsite.com/cover.jpg",
    "cms": "hexo",
    "amount": 5
}
```

### Q: 如何处理图片？

A: 脚本会自动下载并替换文章中的图片链接。图片将保存在每个站点的 `uploads` 目录下。例如：
```
\example.com\uploads\1234567890abcdef.jpg
```

## 贡献指南

1. **Fork** 项目到你的 GitHub 账户。

2. **Clone** 你的 Fork 到本地：

   ```bash
   git clone https://github.com/lopinx/Crawl4GEO.git
   cd Crawl4GEO
   ```
3. **创建** 一个新的分支：

   ```bash
   git checkout -b feature/your-feature
   ```
4. **提交** 你的更改：

   ```bash
   git add .
   git commit -m "Add your feature"
   ```
5. **Push** 到你的分支：

   ```bash
   git push origin feature/your-feature
   ```

6. 打开一个 **Pull Request** 到 `main` 分支。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

希望这个 `README.md` 文档能够帮助用户更好地理解和使用你的项目！如果有任何补充或修改需求，请随时告知。