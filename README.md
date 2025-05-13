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

## 项目概述

`Crawl4GEO` 是一个用于从多个站点爬取文章并生成结构化数据的自动化脚本。该脚本支持多站点爬取，使用 `KeyBERT` 和 `TextRank`, `jiba` 等算法进行关键词提取，并支持将爬取的文章导出为 Markdown 格式，方便集成到各种内容管理系统（CMS/LLM）中。

## 配置说明

### `config.json`

`config.json` 文件包含爬虫的配置信息，包括站点信息、选择器、过滤规则等。以下是一个示例配置：

``` json5
[
    {
        "name": "蝙蝠侠IT（www.batmanit.com）",
        "start_urls": [
            "https://www.batmanit.com/fl-1.html"
        ],
        "selectors": {
            "list": {
                "article": "article.post > header > h2 > a",
                "next": "nav.pagination > a:last-child"
            },
            "article": {
                "title": "h1.article-title",
                "tags": "",
                "excerpt": "",
                "content": "article.article-content",
                "date": "",
                "extras": {} 
            },
            "remove": {
                "tags": [
                    "",
                    ""
                ],
                "regex": [
                    "<script(.*?)><\\/script>",
                    "<p>&nbsp;<\\/p>",
                    "<p><br\\s*\\/?></p>"
                ]
            },
            "filter": {
                "titles": [],
                "words": []
            }
        },
        // 以下为非必须参数
        // CMS类型
        "cms": "hugo",
        // 文章分类
        "categories": [
            ""
        ],
        // 文章作者
        "author": "lopins",
        // 线程数量
        "amount": 100,
        // 图片前缀
        "imgprefix": "https://cdn.lopins.cn/images",
        // 爬虫UA
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        // 是否分词
        "extract": true,
        // 水印参数
        "watermark": {
            "text": "",
            "password": ""
        },
        // 链接样式
        "slugify": {
            "stopwords": [],
            "replacements": [],
            "separator": "-"
        },
        // 分词语库
        "tokenizer": {
            "filter": [],
            "require": []
        }
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

A: 编辑 `config.json` 文件，添加新的站点配置。例如：

``` json5
{
    "name": "蝙蝠侠IT（www.batmanit.com）",
    "start_urls": [
        "https://www.batmanit.com/fl-1.html"
    ],
    "selectors": {
        "list": {
            "article": "article.post > header > h2 > a",
            "next": "nav.pagination > a"
        },
        "article": {
            "title": "h1.article-title",
            "tags": "",
            "excerpt": "",
            "content": "article.article-content",
            "date": "",
            "extras": {} 
        },
        "remove": {
            "tags": [
                "",
                ""
            ],
            "regex": [
                "<script(.*?)><\\/script>",
                "<p>&nbsp;<\\/p>",
                "<p><br\\s*\\/?></p>"
            ]
        },
        "filter": {
            "titles": [],
            "words": []
        }
    },
    // 以下为非必须参数
    // CMS类型
    "cms": "hugo",
    // 文章分类
    "categories": [
        ""
    ],
    // 文章作者
    "author": "lopins",
    // 线程数量
    "amount": 100,
    // 图片前缀
    "imgprefix": "https://cdn.lopins.cn/images",
    // 爬虫UA
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    // 是否分词
    "extract": true,
    // 水印参数
    "watermark": {
        "text": "",
        "password": ""
    },
    // 链接样式
    "slugify": {
        "stopwords": [],
        "replacements": [],
        "separator": "-"
    },
    // 分词语库
    "tokenizer": {
        "filter": [],
        "require": []
    }
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