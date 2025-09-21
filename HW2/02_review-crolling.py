"""
작성자: Eunhwi Lee
날짜: 2025-09-21
주제: Auto-crawl abstracts (PubMed/ArXiv) → LLM 정형화(JSON) → 테이블 저장
"""

import os, re, json, requests, pandas as pd
from urllib.parse import urlencode
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import xml.etree.ElementTree as ET

# ===== 공통 준비 =====
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")
OUT = Path("HW2/outputs")

# ===== 간단 크롤러: PubMed(E-utilities) / ArXiv =====
def fetch_pubmed(query: str, retmax: int = 10) -> str:
    """PubMed E-utilities: esearch → efetch XML(문자열) 반환"""
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        es = requests.get(
            base + "esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
            timeout=20,
        )
        es.raise_for_status()
        ids = es.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""
        ef = requests.get(
            base + "efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=30,
        )
        ef.raise_for_status()
        return ef.text
    except Exception as e:
        print("[WARN] PubMed fetch fail:", e)
        return ""

def fetch_arxiv(query: str, max_results: int = 10) -> str:
    """ArXiv API: Atom feed(문자열) 반환"""
    try:
        url = f"http://export.arxiv.org/api/query?{urlencode({'search_query': query, 'start': 0, 'max_results': max_results})}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print("[WARN] ArXiv fetch fail:", e)
        return ""

# ===== PubMed/ArXiv → 논문 단위 파싱 =====
def parse_pubmed_xml(xml_text: str):
    """PubMed XML → [{'title','year','abstract','citation'}...]"""
    out = []
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out
    for art in root.findall(".//PubmedArticle"):
        title = art.findtext(".//ArticleTitle", default="") or ""
        abstract = " ".join((ab.text or "") for ab in art.findall(".//AbstractText")) or ""
        # 연도 우선순위: Year > MedlineDate(YYYY)
        year = art.findtext(".//Journal/JournalIssue/PubDate/Year", default="")
        if not year:
            md = art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", default="")
            m = re.search(r"\b(19|20)\d{2}\b", md or "")
            year = m.group(0) if m else ""
        fa = art.find(".//AuthorList/Author")
        first_author = ""
        if fa is not None:
            last = fa.findtext("LastName") or ""
            init = fa.findtext("Initials") or ""
            first_author = (last + " " + init).strip()
        cite = f"{first_author} {year}".strip()
        out.append({"title": title, "year": year, "abstract": abstract, "citation": cite})
    return out

def parse_arxiv_atom(atom_text: str):
    """ArXiv Atom → [{'title','year','abstract','citation'}...]"""
    out = []
    if not atom_text:
        return out
    try:
        root = ET.fromstring(atom_text)
    except Exception:
        return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns):
        title = e.findtext("a:title", default="", namespaces=ns) or ""
        summary = e.findtext("a:summary", default="", namespaces=ns) or ""
        updated = e.findtext("a:updated", default="", namespaces=ns) or ""
        m = re.match(r"(\d{4})", updated)
        year = m.group(1) if m else ""
        author_node = e.find("a:author/a:name", ns)
        first_author = (author_node.text if author_node is not None else "").split()[-1]
        cite = f"{first_author} {year}".strip()
        out.append({"title": title, "year": year, "abstract": summary, "citation": cite})
    return out

# ===== 키워드 필터 (addiction + TMS/tDCS + RLDM) =====
RLDM_HINTS = [
    "reward prediction error", "RPE", "model-based", "model free", "model-free",
    "delay discounting", "reward sensitivity", "pavlovian", "exploration", "exploitation"
]
NIBS_HINTS = ["tms", "tdcs", "transcranial magnetic stimulation", "transcranial direct current stimulation"]
ADDICT_HINTS = ["addiction", "substance use", "smoking", "alcohol", "opioid", "cocaine", "methamphetamine", "nicotine", "gambling"]

def contains_any(text: str, keywords):
    t = (text or "").lower()
    return any(k in t for k in keywords)

def keyword_filter(records):
    """addiction + (TMS/tDCS) + RLDM 모두 포함 문서만 유지"""
    keep = []
    for r in records:
        blob = f"{r.get('title','')} {r.get('abstract','')}"
        if contains_any(blob, ADDICT_HINTS) and contains_any(blob, NIBS_HINTS) and contains_any(blob, RLDM_HINTS):
            keep.append(r)
    return keep

# ===== 수집 & 파싱 & 필터 =====
QUERY_PM = '(addiction OR "substance use") AND (TMS OR tDCS) AND (reinforcement learning OR computational)'
QUERY_AX = 'all:(addiction AND (TMS OR tDCS))'

pm_xml = fetch_pubmed(QUERY_PM, retmax=15)
ax_atom = fetch_arxiv(QUERY_AX, max_results=15)

papers = parse_pubmed_xml(pm_xml) + parse_arxiv_atom(ax_atom)
papers = keyword_filter(papers)

# 논문별 블록(최대 12개) → LLM 투입 텍스트
blocks = []
for p in papers[:50]:
    blocks.append(
        "TITLE: {t}\nYEAR: {y}\nABSTRACT: {a}\nCITATION: {c}".format(
            t=p.get("title","").strip(),
            y=p.get("year","").strip(),
            a=p.get("abstract","").strip(),
            c=p.get("citation","").strip(),
        )
    )
docs_text = "\n\n---\n\n".join(blocks)

# ===== 스키마 정의 =====
schema_cols = ["Year", "FirstAuthor", "Title", "Sample", "Paradigm/Task", "RLDM_Construct", "NIBS_Type&Target", "Protocol", "BehavioralOutcome", "ComputationalMeasure", "NeuralMeasure(fMRI/EEG)", "Limitation"]

# TODO: 프롬프트 작성  (기본 틀 유지: prompt → generate_content → print)
prompt = f"""
너는 리뷰 데이터 큐레이터다. 아래 '논문별 블록'에서 **addiction + (TMS|tDCS) + RLDM** 조건을 만족하는 항목만 추출해
다음 스키마의 **JSON 배열**(list of objects)로만 반환하라. (코드펜스 금지) 

스키마 키(문자열):
{schema_cols}

규칙:
- 각 객체는 하나의 논문을 나타낸다.
- 근거 불명확 필드는 ""로 둔다(추정·날조 금지).
- NIBS_Type&Target 예: "tDCS, left dlPFC" / "TMS, vmPFC"
- 반환은 **순수 JSON 배열**만. 설명/마크다운/코드펜스 추가 금지.

논문별 블록:
{docs_text}
"""

# 실행 및 출력
response = model.generate_content(prompt)
print(response.text)

# ===== JSON 파싱 & 저장 =====
txt = (response.text or "").strip()
m = re.search(r'\[.*\]', txt, flags=re.S)
data = json.loads(m.group(0)) if m else []
df = pd.DataFrame(data, columns=schema_cols) if data else pd.DataFrame(columns=schema_cols)

# 저장
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "02_auto_table.csv", index=False)
Path(OUT / "02_auto_table.json").write_text(
    df.to_json(orient="records", force_ascii=False, indent=2),
    encoding="utf-8"
)

# TODO: 결과 분석
"""
실행 결과:
- 여러번 시행시 총 2-3편의 논문이 JSON 형식으로 정리됨. (직접 서치했을 때는 그것보다 많은 논문 존재함 --> 로직에서의 문제 시사)
- 공통적으로 addiction + tDCS/TMS + RLDM 조건을 충족하는 논문만 필터링됨.
- BehavioralOutcome과 Limitation은 구체적으로 기술됨.
- 그러나 ComputationalMeasure, NeuralMeasure 필드는 모두 빈 칸 → 실제 초록에서 추출하거나 태스크 기반으로 추론하지 않음.

개선 방향:
- 논문 크롤링이 제대로 되지 않는 것 같음. 관련하여 코드 수정 필요함
- RLDM 키워드가 초록에 없더라도 태스크명 기반 매핑 규칙을 코드에 직접 넣어 RLDM_Construct를 채울 수 있도록 보완.
- ComputationalMeasure: 강화학습 모델 적용 여부, 파라미터(learning rate, RPE 등) 언급 여부를 탐지.
- NeuralMeasure: fMRI/EEG 단어 탐색하여 기본값이라도 채우도록 개선.
"""