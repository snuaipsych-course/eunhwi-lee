"""
작성자: Eunhwi Lee
날짜: 2025-09-21
주제: 리뷰 페이퍼 scope 및 outline 작성
"""

import os, json, glob
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Config ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
assert API_KEY, "GOOGLE_API_KEY missing in .env"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
OUT = Path("HW2/outputs")

# ===== 유틸: PDF 일부 로드 (없으면 빈 문자열) =====
def read_pdf_snippet(pattern, max_chars=800):
    try:
        import fitz  # pip install pymupdf
    except ImportError:
        return ""
    buf = []
    for p in glob.glob(pattern, recursive=True):
        try:
            doc = fitz.open(p)
            txt = "\n".join(pg.get_text() for pg in doc); doc.close()
            buf.append(f"\n--- {os.path.basename(p)} ---\n{txt}")
        except Exception as e:
            print("[WARN] PDF read fail:", e)
    return ("\n".join(buf))[:max_chars]

HW0 = read_pdf_snippet("HW0/**/*.pdf")
STYLE = read_pdf_snippet("HW1/**/*.pdf") + "\n" + read_pdf_snippet("HW2/**/*.pdf")

# TODO: 프롬프트 작성
prompt = f"""
당신은 computational psychiatry 연구자의 리뷰페이퍼 코치입니다. 
리뷰 주제는 addiction을 대상으로 한 brain stimulation(TMS/tDCS) 연구에서 
reinforcement learning & decision-making(RLDM) 기반 computational psychiatry 지표가 어떻게 활용되었는지입니다. 
나는 systematic narrative review를 작성하려 하며, 연구자의 시각은 'construct-driven' 접근(예: RPE, model-based vs model-free, reward sensitivity 등)을 
translational하게 brain stimulation 맥락에 연결하는 데 있습니다.

## Task
- 주제: Computational psychiatry × brain stimulation(TMS/tDCS) × addiction — systematic narrative review.
- 산출물:
  1) 스코프(Scoping, 120–160단어): RLDM construct들을 중독 연구에서 어떻게 operationalize했는지, brain stimulation 연구에서 어떤 방식으로 outcome 또는 mediator로 다뤄졌는지를 요약. 
     Scoping은 'translational relevance'를 강조해야 함 (예: parameter tracking for mechanism, computational phenotyping for prediction).
  2) 포함/제외 기준(Inclusion/Exclusion, 불릿 6–10개): 연구 디자인(임상/실험), 집단(임상군/건강대조군), 중재(tDCS/TMS), 결과(behavioral + computational + neural), 
     언어·연도·데이터타입(인간 대상 연구, peer-reviewed 논문) 등 systematic review 관점에서 구체화.
  3) 목차(Outline): H2(3–5개)와 각 H4(2–4개). 구조는 Introduction, Methods, Results, Discussion(한계·translational 함의·미래 방향) 흐름을 반영.

## 참고(요약/인용 금지)
<<HW0_CONTEXT>>
{HW0}
<</HW0_CONTEXT>>

<<STYLE_GUIDE>>
{STYLE}
<</STYLE_GUIDE>>

- 출력 형식: H2/H4 + 불릿
"""

# 실행 및 출력
response = model.generate_content(prompt)
print(response.text)

# TODO: 결과 분석
"""
실행 결과:
모델은 addiction 맥락의 NIBS(TMS/tDCS) × computational psychiatry 리뷰 구조를 잘 따라가며,
- Scoping에서는 RLDM construct(RPE, model-based/model-free, reward sensitivity)를 translational 관점에서 정리
- Inclusion/Exclusion은 연구 디자인, 중재, 대상, 변수, 연도 등을 체계적으로 제시
- Outline은 Introduction–Methods–Results–Discussion으로 조직, Results를 construct별 synthesis로 구체화
전반적으로 systematic review 형식을 충실히 반영한 출력이 생성됨

개선 방향:
- 본 출력을 기반으로 실제 데이터 크롤링을 활용하여 리뷰할 논문을 리스트업하는 형태로 chain 구성 가능
"""

(OUT / "01_response.md").write_text(response.text, encoding="utf-8")
