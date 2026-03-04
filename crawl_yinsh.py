import os
import re
import time
import json
import string
import requests
import pandas as pd
from tqdm import tqdm
from lxml import etree
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# 0) NLTK stopwords 준비
# ----------------------------
def ensure_nltk():
    import nltk
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words("english")
    except Exception:
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

ensure_nltk()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# 1) 설정
# ----------------------------
GAME_ID = 7854  # YINSH
BASE = "https://boardgamegeek.com/xmlapi2"
OUTDIR = "/crdata"
os.makedirs(OUTDIR, exist_ok=True)

REQUEST_SLEEP_SEC = 5.0
TIMEOUT = 30

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

session = requests.Session()
session.headers.update({"User-Agent": UA})

def api_get(path, params=None, sleep_sec=REQUEST_SLEEP_SEC, max_retry=6):
    url = f"{BASE}/{path}"
    params = params or {}
    for attempt in range(max_retry):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            if r.status_code in (500, 503, 429):
                wait = sleep_sec * (2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(sleep_sec)
            return r.text
        except requests.RequestException:
            wait = sleep_sec * (2 ** attempt)
            if attempt == max_retry - 1:
                raise
            time.sleep(wait)

def parse_xml(xml_text):
    return etree.fromstring(xml_text.encode("utf-8"))

# ----------------------------
# 2) forumlist -> forum ids
# ----------------------------
def get_forum_list_for_game(game_id: int):
    xml = api_get("forumlist", {"id": str(game_id), "type": "thing"})
    root = parse_xml(xml)
    forums = []
    for f in root.xpath(".//forum"):
        forums.append({
            "forum_id": int(f.get("id")),
            "title": f.get("title", ""),
            "description": f.get("description", ""),
            "numthreads": int(f.get("numthreads") or 0),
            "numposts": int(f.get("numposts") or 0),
        })
    return pd.DataFrame(forums).sort_values(["numposts", "numthreads"], ascending=False)

# ----------------------------
# 3) forum -> thread ids
# ----------------------------
def get_threads_in_forum(forum_id: int):
    all_threads = []
    page = 1
    while True:
        xml = api_get("forum", {"id": str(forum_id), "page": str(page)})
        root = parse_xml(xml)

        threads = root.xpath(".//thread")
        if not threads:
            break

        for t in threads:
            all_threads.append({
                "forum_id": forum_id,
                "thread_id": int(t.get("id")),
                "subject": t.get("subject", ""),
                "author": t.get("author", ""),
                "numarticles": int(t.get("numarticles") or 0),
                "postdate": t.get("postdate", ""),
                "lastpostdate": t.get("lastpostdate", ""),
            })

        if len(threads) < 50:
            break
        page += 1

    return pd.DataFrame(all_threads)

# ----------------------------
# 4) thread -> articles(댓글 포함)
# ----------------------------
def get_thread_articles(thread_id: int, count=1000):
    xml = api_get("thread", {"id": str(thread_id), "count": str(count)})
    root = parse_xml(xml)

    thread_title = root.xpath("string(.//thread/@subject)") or ""
    articles = []
    for a in root.xpath(".//article"):
        body = a.xpath("string(./body)") or ""
        articles.append({
            "thread_id": thread_id,
            "thread_title": thread_title,
            "article_id": int(a.get("id") or 0),
            "username": a.get("username", ""),
            "postdate": a.get("postdate", ""),
            "subject": a.get("subject", ""),
            "body": body,
        })
    return articles

# ----------------------------
# 5) 텍스트 전처리/토큰화
# ----------------------------
lemmatizer = WordNetLemmatizer()

CUSTOM_STOPWORDS = set([
    "bgg", "boardgamegeek", "thread", "post", "quote", "quoted", "edit",
    "yinsh", "game", "games", "player", "players",
])

STOPWORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-zA-Z0-9\- ]+")

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = NON_WORD_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def tokenize(s: str):
    s = normalize_text(s)
    tokens = []
    for tok in s.split():
        tok = tok.strip(string.punctuation)
        if not tok or tok.isdigit() or len(tok) <= 2:
            continue
        if tok in STOPWORDS:
            continue
        tok = lemmatizer.lemmatize(tok)
        if tok in STOPWORDS or len(tok) <= 2:
            continue
        tokens.append(tok)
    return tokens

def build_unigram_counter(texts):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c

# ----------------------------
# 6) WordCloud 저장+표시
# ----------------------------
def save_and_show_wordcloud(freq: dict, outpath: str, title: str):
    wc = WordCloud(
        width=1400, height=800,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    # 저장
    plt.savefig(outpath, dpi=200)

    # 화면 출력(가능한 환경에서)
    plt.show()
    plt.close()

# ----------------------------
# 7) 메인
# ----------------------------
def main():
    # (1) 포럼 목록
    forums_df = get_forum_list_for_game(GAME_ID)
    forums_df.to_csv(os.path.join(OUTDIR, "forums.csv"), index=False, encoding="utf-8-sig")
    print(f"[Forums] {len(forums_df)} forums")

    # (2) 스레드 목록
    all_threads = []
    for _, row in forums_df.iterrows():
        fid = int(row["forum_id"])
        tdf = get_threads_in_forum(fid)
        if not tdf.empty:
            all_threads.append(tdf)

    if not all_threads:
        print("No threads found.")
        return

    threads_df = pd.concat(all_threads, ignore_index=True)
    threads_df.to_csv(os.path.join(OUTDIR, "threads.csv"), index=False, encoding="utf-8-sig")
    print(f"[Threads] {len(threads_df)} threads")

    # (3) 댓글 포함 게시물 수집
    all_articles = []
    for tid in tqdm(threads_df["thread_id"].unique().tolist(), desc="Fetching threads"):
        try:
            all_articles.extend(get_thread_articles(int(tid)))
        except Exception as e:
            with open(os.path.join(OUTDIR, "errors.log"), "a", encoding="utf-8") as f:
                f.write(f"thread_id={tid} error={repr(e)}\n")

    articles_df = pd.DataFrame(all_articles)
    articles_df.to_csv(os.path.join(OUTDIR, "articles.csv"), index=False, encoding="utf-8-sig")
    print(f"[Articles] {len(articles_df)} articles")

    # (4) 분석 텍스트: subject + body
    docs = (articles_df["subject"].fillna("") + " " + articles_df["body"].fillna("")).tolist()

    # (5) 키워드(빈도) 산출
    uni = build_unigram_counter(docs)
    top5 = uni.most_common(5)

    # 저장(원하면 나중에 확인)
    with open(os.path.join(OUTDIR, "top_unigrams.json"), "w", encoding="utf-8") as f:
        json.dump(uni.most_common(200), f, ensure_ascii=False, indent=2)

    # ✅ 콘솔 출력: 키워드 5개만
    print("\nTop 5 keywords (frequency):")
    for i, (w, c) in enumerate(top5, 1):
        print(f"{i}) {w}\t{c}")

    # (6) WordCloud 저장 + 출력
    wc_path = os.path.join(OUTDIR, "wordcloud_unigram.png")
    save_and_show_wordcloud(dict(uni.most_common(200)), wc_path, "YINSH Forum Keywords (Unigram)")
    print(f"\n[Saved] WordCloud -> {wc_path}")

    print(f"\nDone. All outputs saved under: {OUTDIR}")

if __name__ == "__main__":
    main()