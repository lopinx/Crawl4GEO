# -*- coding: utf-8 -*-
__author__ = "https://github.com/lopinx"
# ======================================================================================================================
# 配置文件： pyproject.toml
# ======================================================================================================================
# [[tool.uv.index]]
# url = "https://mirrors.huaweicloud.com/repository/pypi/simple"
# default = true
# ======================================================================================================================
# 安装依赖包： uv add markdownify ... ...
# 安装依赖包： uv pip install -r requirements.txt
# 导出依赖包： uv pip freeze | uv pip compile - -o requirements.txt
# uv add aiofiles aiosqlite "httpx[http2,http3]" keybert scikit-learn jieba nltk toml lxml bs4 markdown markdownify pillow python-slugify pypinyin rank_bm25
# ======================================================================================================================
# 关键词提取：Rake、Yake、Keybert 和 Textrank
# 百度：Textrank + jieba
# 谷歌：Keybert
# ======================================================================================================================
import asyncio
import hashlib
import json
import logging
import mimetypes
import os.path
import re
import sys
import uuid
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from itertools import product
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import (unquote, urljoin, urlparse)

import aiofiles
import aiosqlite
import jieba
import jieba.analyse
import markdown
import nltk
import numpy as np
import tomlkit
from bs4 import BeautifulSoup, NavigableString
from httpx import AsyncClient, HTTPError, Response
from keybert import KeyBERT
from markdownify import markdownify
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.util import everygrams
from PIL import Image
from pypinyin import Style, lazy_pinyin
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from slugify import slugify

# ======================================================================================================================
"""全局设置"""
# 当前工作目录
WorkDIR = Path(__file__).resolve().parent
sites = json.load(open(WorkDIR/"config.json", 'r', encoding='utf-8'))
# 下载分词库数据（首次运行需要）
try:
    nltk.corpus.stopwords.words()
except LookupError:
    nltk.download('stopwords')
try:
    nltk.sent_tokenize("Test sentence")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
# 远程 https://res.cdn.issem.cn/ChineseStopWords.txt, 并将内容转换为列表
cn_stopk, en_stopk = [[*map(str.strip, filter(str.strip, (WorkDIR / f"{lang}StopWords.txt").open(encoding='utf-8')))] for lang in ('Chinese', 'English')]
# ======================================================================================================================

class ArticleSpider():
    """多站点文章爬虫类"""
    def __init__(self, site: Dict) -> None:
        self.site = site
        self.logger = self._setup_logging()
        self.client = AsyncClient()
    
    # 设置日志记录
    def _setup_logging(self) -> Union[logging.Logger, None]:
        domain = urlparse(self.site['start_urls'][0]).netloc
        log_path = Path(WorkDIR) / domain / "bot.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(domain)

    # 设置数据链接
    async def _setup_db(self) -> None:
        domain = urlparse(self.site['start_urls'][0]).netloc
        db_path = Path(WorkDIR) / domain / "log.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(db_path)
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                tags TEXT,
                excerpt TEXT,
                content TEXT,
                date TEXT,
                extras JSONB,
                pictures TEXT,
                publish BOOLEAN DEFAULT 0
            )
            """
        )
        await self.conn.commit()

    # 抓取远程数据
    async def _fetch_data(self, url: str, filename: str | None) -> Tuple[bool, Response | Union[str, Path, None]]:
        _parses = urlparse(self.site['start_urls'][0])
        _pn = unquote(PurePath(urlparse(url).path.rstrip('/')).name) if filename else None
        _headers = {
            'User-Agent': self.site.get('user_agent', "lopins.cn/0.1 Fetcher, support@lopins.cn"),
            'Host': _parses.netloc,
            'Accept': '*/*',
            'Content-Type': mimetypes.guess_type(_pn)[0] if filename else 'text/html',
            **(dict(ContentDisposition=f'inline;filename="{_pn}"') if filename else {}),
        }
        _result = False, None
        # 尝试抓取数据
        try:
            resp = await self.client.get(
                url, 
                headers={**_headers, 'Referer': f'{_parses.scheme}://{_parses.netloc}'},
                follow_redirects=True,
                timeout=60.0
            )
            resp.raise_for_status()
            _result = True, resp
        except:
            pass
        # 本地保存图片
        if filename and _result[0]:
            image_path = Path(WorkDIR) / _parses.netloc / "uploads" / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                loop = asyncio.get_event_loop()
                img_bytes = BytesIO(_result[1].content)
                with await loop.run_in_executor(None, Image.open, img_bytes) as img:
                    if img.format.upper() not in ["JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "ICO", "SVG"]:
                        raise ValueError("不支持的图片格式")
                async with aiofiles.open(image_path, mode='wb') as f:
                    await f.write(_result[1].content)
                if image_path.exists():
                    _result = True, image_path
            except HTTPError as e:
                self.logger.error(f"图片失败: {url} - {str(e)}")
            except:
                pass
        return _result

    # 提取元关键词
    async def _extract_keywords(self, content: str, require_words: List[str]) -> List[str]:
        if not content: return []
        # 处理成Markdown格式
        try:
            html = markdown.markdown(content)
            if not (('<' in html and '>' in html) and html.strip() != content.strip()):
                content = markdownify(content)
        except Exception:
            content = markdownify(content)
        # 清理非法字符（保留中英文、数字、常见符号）
        # 需要匹配英文单词、中文、专有英文缩写（如 NASA、U.S.A.）以及特定符号
        content = re.sub(
            r'[^a-zA-Z0-9\u4e00-\u9fa5\s\-.\'@#$%&*+/:;=?~(){}$`_、。，《》？！“”‘’（）—…]',
            ' ',                                                # 用空格替代非法字符
            content,                                            # 替换目标字符串
            flags=re.UNICODE
        ).strip()
        # 判断文本语言（中文/英文）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in content
        )
        # 可选停用词
        if not cn_lang:
            stop_words = set(stopwords.words('english')).union(en_stopk).union(list(set(self.site['tokenizer']['filter'])))
        else:
            stop_words = set(stopwords.words('chinese')).union(cn_stopk).union(list(set(self.site['tokenizer']['filter'])))
        # ===============================================================================================================
        # Keybert 算法（英文） / TextRank + jieba 算法（中文）
        # ===============================================================================================================
        if not cn_lang:
            # 英文处理：KeyBERT + 语义优先
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(
                content.lower(),
                keyphrase_ngram_range=(1, 6),
                stop_words=stop_words,
                use_mmr=True,
                diversity=0.5
            )
            vectorizer = TfidfVectorizer(ngram_range=(1, 6))
            tfidf = vectorizer.fit_transform([content])
            vocab = vectorizer.vocabulary_  # 获取词汇表
            # 综合评分：BERT置信度 × TF-IDF
            scores = {
                word: score * tfidf[0, vocab.get(word, -1)]
                for word, score in keywords
                if word in vocab
            }
        else:
            # 中文处理：TextRank + 词密度
            # 将用户词逐个添加到jieba词典（确保短语不被拆分）
            list(map(jieba.add_word, require_words))
            keywords = [word for word in jieba.cut(content) if word not in stop_words or word in require_words]
            vectorizer = TfidfVectorizer(ngram_range=(1, 6), token_pattern=r'[^\s]+')
            tfidf_matrix = vectorizer.fit_transform([' '.join(keywords)])
            vocab = vectorizer.vocabulary_  
            tfidf = tfidf_matrix.toarray()[0]
            # TextRank权重（允许短语）
            text_rank = {k: v for k, v in jieba.analyse.extract_tags(' '.join(keywords), topK=500, withWeight=True, allowPOS=())}
            scores = {
                word: text_rank.get(word, 0) * tfidf[vocab[word]] 
                for word in keywords 
                if word in vocab
            }
        # 强制合并用户词典中的词（中英文统一处理）
        for key in require_words:
            word = key.lower()
            if word in vocab and word not in scores:
                scores[word] = tfidf[vocab[word]] if cn_lang else 1.0 * tfidf[0, vocab[word]]
        # 强制包含用户提供词（统一逻辑）
        max_score = max(scores.values(), default=0)
        for req_word in (r.lower() for r in require_words):
            current = scores.get(req_word, 0)
            scores[req_word] = current * 2 if current else max_score * 2 + 1

        # 生成排序后的关键词列表
        t_k = sorted(scores.keys(), key=lambda k: (-scores[k], -len(k)))
        # 过滤不符合条件的关键词
        filter_pattern = re.compile(r'^[\W_]+$|^\d+(?:\.\d+)?%?$|^\d+[eE][+-]?\d+$')
        return [k for k in t_k if len(k) >= 2 and not filter_pattern.fullmatch(k)]
        # ===============================================================================================================

    # 提取元摘要
    async def _extract_excerpt(self, content: str, length: int = 3) -> str:
        # 先将markdown格式转换为纯文本格式
        sentences = sent_tokenize(markdownify(content))
        # 判断文本语言（中文/英文）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in content
        )
        # 分句处理
        if not cn_lang:
            sentences = sent_tokenize(content)
            stop_words = set(stopwords.words('english')).union(en_stopk)
        else:
            sentences = [s.strip() for s in re.split(r'[。！？\.\!\?]\s*', content) if s.strip()]
            stop_words = set(stopwords.words('chinese')).union(cn_stopk)
        # 分词并去除停用词
        _sents = []
        for sent in sentences:
            if not cn_lang:
                tokens = [word for word in word_tokenize(sent.lower()) if word not in stop_words]
            else:
                tokens = [word for word in jieba.cut(sent) if word not in stop_words]
            _sents.append(tokens)
        # 计算 BM25
        bm25 = BM25Okapi(_sents)
        # 计算每个句子的得分
        scores = []
        for query in _sents:
            scores.append(bm25.get_scores(query).mean())  # 使用平均得分
        # 获取得分最高的句子索引并返回摘要
        excerpt = ' '.join([sentences[i] for i in sorted(np.argsort(scores)[::-1][:length])])
        return excerpt

    # 修复内容图片
    async def _fix_images(self, link: str, title: str, html: BeautifulSoup) -> Tuple[BeautifulSoup, List, List]:
        # 处理 <picture> 标签
        for picture in html.select('picture'):
            img = picture.find('img')
            if img:
                picture.replace_with(img)

        local, thumb, img_tags = [], [], html.select('img')

        # 修复远程图片调用解析器隐藏的图片真实地址
        imgig_parses = [f'/{p}.php?{q}=' for p, q in product(['img','image','link','pic','api'], ['url','img','image','pic'])]
        imgig_domains = [
            '.baidu.com',
            '.byteimg.com',
            '.zhimg.com',
            'user-gold-cdn.xitu.io',
            'upload-images.jianshu.io',
            'img-blog.csdnimg.cn',
            'nimg.ws.126.net',
            '.360buyimg.com',
            '.sinaimg.cn',
            'user-images.githubusercontent.com',
            '.qhimg.com',
            '.alicdn.com',
            '.tbcdn.cn',
            '.cn.bing.net',
            '.sogoucdn.com',
            '.360tres.com'
        ]

        # 处理所有图片标签
        for img_tag in img_tags:
            # 跳过无src的标签和Base64图片
            if not (src := img_tag.get('src')) or 'data:image/' in (img_tag.get('data-original') or src): continue
            # 获取原始图片URL
            img_url = img_tag.get('data-original') or src
            # 修复相对路径
            real_url = urljoin(link, img_url)
            # 获取真实URL
            real_url = next((real_url.split(_)[1] for _ in imgig_parses if _ in real_url), real_url)
            # 检查远程域名
            if (_u := urlparse(real_url)) and any(_u.netloc.endswith(_) for _ in imgig_domains): continue
            # 判断修复后的链接是否可以正常访问,如果不能访问，则删除图片对象并跳过该图片标签
            try:
                # 下载远程图片
                filename = f"{hashlib.md5(real_url.encode()).hexdigest()}.jpg"
                _realurl = await self._fetch_data(real_url, filename)
                local.append(_realurl[1]) if _realurl[0] else None
                # 替换内容URL
                img_tag['data-original' if 'data-original' in img_tag.attrs else 'src'] = f"{self.site['imgprefix']}/{filename}"
                # 设置元数据
                img_tag['alt'] = img_tag['title'] = title
            except Exception as e:
                img_tag.decompose()
                continue

        # 获取缩略图
        thumb = list({_['src'] for _ in html.select('img[src]')}) or []
        # 重新返回处理后字符串对象（替换后图片链接内容）
        return html, local, thumb

    # 提取文章信息
    async def _parse_article(self, link: str, html: str) -> Tuple[Dict, List, List]:
        require_words = list(set(self.site['tokenizer']['require']))
        soup = BeautifulSoup(html, 'lxml')
        """ 获取想要的标签属性 """
        # 标题(纯文本)
        try:
            _title = soup.select_one(self.site['selectors']['article']['title']).text.strip().lstrip("标题：")
            # 只保留需要的特定标题信息
            if (tfs := self.site['selectors']['filter']['titles']) and any(_ in _title for _ in tfs):
                return
            if (sfw := self.site['selectors']['filter']['words']) and any(_ in _title for _ in sfw):
                return
        except:
            return

        # 内容（BeautifulSoup对象）
        # =========================================================================================================
        try:
            _html = soup.select_one(self.site['selectors']['article']['content'])
            # 过滤违禁信息
            if (sfw := self.site['selectors']['filter']['words']) and any(_ in _html.text for _ in sfw):
                return
            desc_preg = r'^.*?【?(?:本文|文章|本篇|全文|前言)\s*[，,]?\s*(?:简介|摘要|概述|导读|描述)】?[：:]?'
            if (p := re.compile(desc_preg, re.DOTALL)) and (_p := _html.find(lambda _t: _t.text and p.match(_t.text))):
                _p.string = re.sub(p, '', _p.text, 1)
        except:
            return
        # 移除不需要的元素【指定选择器+正则匹配】
        # 步骤1：移除符合指定选择器的页面元素
        if srt := self.site['selectors']['remove']['tags']:
            for selector in srt:
                try:
                    for element in _html.select(selector):
                        element.decompose()
                except Exception as e:
                    pass
        # 步骤2：移除符合正则表达式的文本内容
        if srr := self.site['selectors']['remove']['regex']:
            for pattern in srr:
                try:
                    for text in _html.find_all(string=lambda t: re.match(pattern, str(t))):
                        text.extract()
                except Exception as e:
                    pass
        # =========================================================================================================

        # 提取文章摘要
        try:
            _excerpt = soup.select_one(f"{self.site['selectors']['article']['excerpt']}").text.strip()
        except:
            try:
                _excerpt = soup.find('meta', name='description').get('content', '') or ''
            except:
                # 过滤掉包含图片的段落后获取摘要
                if (_ps := [p for p in _html.find_all('p') if p.find('img') is None]):
                    _ft = _ps[0].get_text()
                    _excerpt = e if (len(e := (_ft or (_ps[1].get_text() if len(_ps) > 1 else ""))) > 60) else ""
                else:
                    _excerpt = await self._extract_excerpt(_html.text, 5)
        

        # 提取文章标签
        if self.site["extract"]:
            _tags = await self._extract_keywords(_html.text, require_words)
        else:
            try:
                _tags = [_t for _ in soup.select(self.site['selectors']['article']['tags']) if (_t := _.text.strip())]
            except:
                try:
                    _tags = [_t for _ in (soup.find('meta', name='keywords').get('content', '') or '').split(',') if (_t := _.strip())]
                except:
                    _tags = []
            # 标签过少，或者强制分词
            if any('(' in _ for _ in _tags):
                _tags = [re.sub(r'[^\w\s]', '', _.replace("标签：", "")).strip() for _t in _tags for _ in re.split(r'[()]', _t) if _]
            if not _tags or any(_.startswith('标签：') for _ in _tags) or any(' ' in _ for _ in _tags):
                _tags = await self._extract_keywords(_html.text, require_words)
            _tags = [_ for _ in _tags if re.search(r'[\u4e00-\u9fa5a-zA-Z]', _)]

        # 提取文章其他
        _extras = {}
        for key, selector in self.site['selectors']['article']['extras'].items():
            try:
                value = soup.select_one(selector).get_text()
            except AttributeError:
                value = 0  # 或者设置为其他默认值
            _extras[key] = value

        # 提取文章日期
        tzinfo = timezone(timedelta(hours=8))
        try:
            _pubdate = soup.select_one(self.site['selectors']['article']['date']).get_text()
            _pubdate = re.sub(r'上午|下午',lambda m:'AM'if m.group()=='上午'else'PM',_pubdate)
            _pubdate = re.sub(r'[^\d:/APM ]',' ',_pubdate).strip().split()
            if len(_pubdate)>=3: _pubdate[1:3] = [p.zfill(2) for p in _pubdate[1:3]]
            fmt = '%Y %m %d %I:%M%p' if 'AM' in _pubdate or 'PM' in _pubdate else '%Y %m %d %H:%M'
            _date = datetime.strptime(' '.join(_pubdate),fmt).replace(tzinfo=tzinfo).strftime('%Y-%m-%dT%H:%M:%S%z')
        except: 
            # 当前时间
            _date = datetime.now().replace(tzinfo=tzinfo).strftime('%Y-%m-%dT%H:%M:%S%z')
        
        # 获取想要的图片文件
        try:
            # 下载和替换图片链接（BeautifulSoup对象）
            _html, _local, _thumb = await self._fix_images(link, _title, _html)
        except Exception as e:
            self.logger.error(f'{link}，获取图片失败，{str(e)}')

        # 增加版权信息
        if t_mark := self.site['watermark']['text']:
            _copyright = "".join(f"{ord(c):03d}\u200B" for c in t_mark)[:-1]
            if (end_tag := _html.find_all(recursive=False)[-1] if _html.contents else None):
                end_tag.insert_after(NavigableString(_copyright))

        # print(repr(_text))
        return {
            'title': _title,
            'tags': _tags,
            'excerpt': _excerpt,
            'content': str(_html),
            'extras': _extras,
            'date': _date,
            'local': _local,
            'thumb': _thumb
        }

    # 文章处理流程
    async def _process_article(self, url: str) -> Optional[bool]:
        # 查询处理状态
        async with self.conn.cursor() as cursor:
            await cursor.execute("SELECT publish FROM articles WHERE url=?", (url,))
            row = await cursor.fetchone()
            if row and row[0]:
                return
        # 抓取网页源码
        html = await self._fetch_data(url, None)
        if html is None or not html[0]:
            return
        # 解析文章数据
        data = await self._parse_article(url, html[1].text.strip())
        if data is None or not data:
            return
        # 导出文档并存库
        _ftag = await self._export_markdown({**data, 'url': url}),
        if _ftag:
            async with self.conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT OR REPLACE INTO articles "
                    "(url, title, tags, excerpt, content, date, extras, pictures, publish) "
                    "VALUES (?, ?, json(?), ?, ?, ?, json(?), json(?), 1)",
                    (
                        url,
                        data['title'],
                        json.dumps(data['tags']),
                        data['excerpt'],
                        data['content'],
                        data['date'],
                        json.dumps(data['extras']),
                        json.dumps(data['thumb']),
                    )
                )
                await self.conn.commit()
        else:
            self.logger.info('No new articles found.')

    # 爬取文章列表
    async def _crawl_lists(self) -> List[str]:
        links = []
        # 从起始页开启爬取
        for start in self.site['start_urls']:
            current = start
            while current:
                html = await self._fetch_data(current, None)
                if not html[0]:
                    break
                # 页面所有文章链接
                soup = BeautifulSoup(html[1].text.strip(), 'lxml')
                article = soup.select(self.site['selectors']['list']['article'])
                links.extend([urljoin(current, a['href']) for a in article])
                # 自动遍历寻找下页
                next_link = soup.select_one(self.site['selectors']['list']['next'])
                current = (next_link and urljoin(current, next_link['href'])) or None
        return links

    # 生成CMS文章
    async def _export_markdown(self, data: Dict) -> Optional[bool]:
        # 生成Front Matter
        separator = (sep := self.site['slugify']['separator']) or '-'
        # 将中文部分转换为拼音（保留英文和数字）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in data.get('title', '')
        )
        if cn_lang:
            _title = '-'.join(lazy_pinyin(data.get('title', ''), style=Style.NORMAL, strict=False))
        else:
            _title = data.get('title', '')
        urlname = slugify(
            _title,
            separator=separator,
            lowercase=True,
            regex_pattern=None,
            word_boundary=True,
            stopwords=self.site.get('slugify', {}).get('stopwords', []),
            replacements=self.site.get('slugify', {}).get('replacements', [])
        )
        # 创建 TOML 文档对象
        doc = tomlkit.document()
        # 基础字段
        doc["title"] = data.get('title')
        doc["date"] = data.get('date')
        doc["tags"] = data['tags'][:5]
        doc["keywords"] = data['tags'][:5]
        doc["description"] = data.get('excerpt') or ""
        doc["categories"] = self.site['categories']
        doc["author"] = self.site.get('author') or "lopins"
        doc["cover"] = data.get('cover') or ""
        doc["pictures"] = data.get('thumb') or []
        doc["hiddenFromHomePage"] = False
        doc["readingTime"] = True
        doc["hideComments"] = True
        doc["isCJKLanguage"] = True
        doc["slug"] = urlname
        # 处理扩展字段添加到文档
        _extras = {}
        for k, v in data.get('extras', {}).items():
            if isinstance(v, str):
                v = v.strip().strip('(（）)')
                p = int(v) if v.isdigit() else v
            else:
                p = v
            _extras[k] = p if isinstance(p, int) else f'{json.dumps(p,ensure_ascii=False)[1:-1]}'
        for key, value in _extras.items():
            doc[key] = value
        # 文章状态
        doc["draft"] = False
        # 序列化为 TOML 字符串（保留格式）
        front_matter_block = f"+++\n{tomlkit.dumps(doc).strip()}\n+++"

        if self.site['cms'] == 'hexo':
            content = front_matter_block.strip()[3:-3].strip()  # 直接去除+++分隔符
            lines = content.split('\n')
            _yaml = []
            for line in lines:
                line = line.strip()
                if not line: continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    # 处理列表项（移除方括号并分割）
                    items = [i.strip().strip("'\"") for i in val[1:-1].split(',')]
                    _yaml.append(f"{key}:")
                    _yaml.extend(f"  - {item}" for item in items if item)
                elif val in ('true', 'false'):
                    _yaml.append(f"{key}: {val}")
                else:
                    _yaml.append(f"{key}: {val.strip('\"')}")
            front_matter_block = f"---\n{'\n'.join(_yaml)}\n---"
        try:
            domain = urlparse(self.site['start_urls'][0]).netloc
            doc_name = f"{re.sub(r'\D', '', data.get('date'))}-{uuid.uuid4()}.md"
            file_path = Path(WorkDIR) / domain / "article" / doc_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(f"{front_matter_block}\n\n{markdownify(data['content'])}")
            file_size = await asyncio.to_thread(lambda: file_path.stat().st_size)
            return True if file_path.exists() and file_size > 0 else False
        except Exception as e:
            self.logger.error(f"保存文件 {file_path} 失败: {str(e)}")
            return False    

    async def run(self) -> None:
        await self._setup_db()

        _list = asyncio.create_task(self._crawl_lists())
        newurls = await _list  # 获取新的文章链接列表

        semaphore = asyncio.Semaphore(self.site['amount'])
        async def _limit_tasks(task):
            async with semaphore:
                return await task

        _exists = set()
        async with self.conn.cursor() as cursor:
            await cursor.execute("SELECT url FROM articles WHERE publish=1")
            _exists = {_[0] for _ in await cursor.fetchall()}

        tasks = [_limit_tasks(self._process_article(_)) for _ in newurls if _ not in _exists]
        await asyncio.gather(*tasks)

        await self.conn.close()

async def main(sites) -> None:
    await asyncio.gather(*[ArticleSpider(_).run() for _ in sites])

if __name__ == "__main__":
    asyncio.run(main(sites))