"""
작성자: Eunhwi Lee
날짜: 2025-09-21
주제: 크롤링된 논문 요약 바탕으로 리뷰 draft 작성
"""

import os, json, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ===== 준비 =====
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")
OUT = Path("HW2/outputs")

# 표 로드 (02 단계 산출)
table_path = OUT/"02_auto_table.json"
records = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else []
df = pd.DataFrame(records)

# TODO: 프롬프트 작성
prompt = f"""
너는 리뷰 작성 보조자다. 한국어 + 핵심 용어 영문 병기, H2/H4+불릿

## 입력 테이블(요약)
{df.to_markdown(index=False)[:5500]}

## Task
1) Synthesis: Construct/NIBS/Outcome 기준 공통점·차이점(각 3–5 불릿)
2) Critical Appraisal: 방법론 한계 4–6개 + 개선책
3) Future Directions:
   - parameter tracking으로 기전 규명(3–4 불릿)
   - computational phenotyping 기반 개인맞춤 치료(3–4 불릿)
4) Draft(≤400단어): Introduction/Methods scope/Results(표 요약)/Discussion/Conclusion

## 제약
- 표에 없는 사실은 작성하지 말 것(추정/날조 금지)
- HW0/1/2는 글쓰기 스타일 참고만(내용 인용 금지)
"""

# 실행 및 출력
response = model.generate_content(prompt)
print(response.text)

# TODO: 결과 분석
"""
실행 결과:
- Introduction → Methods 범위 → Results 요약(테이블 포함) → Critical Appraisal → Future Directions → Conclusion 의 리뷰페이퍼 전형적 흐름이 잘 구현됨.
- 데이터 반영: 직전 JSON에서 얻어진 논문을 충실히 서론–방법–결과에 반영함.
- 테이블 요약: Construct / NIBS / Outcome 기준으로 비교 가능하게 제시됨.
- 비판적 평가: 일반화 가능성, 세션 수, 표본 크기, 신경생리학적 측정 부재 등 한계가 구체적으로 나열됨.
- 미래 방향: computational parameter tracking, neuroimaging 통합, computational phenotyping 기반 individualized treatment까지 연구 관심사와 정확히 연결됨.
- 톤/구성: 리뷰페이퍼 형식과 일관되며, narrative review 성격에 맞는 포맷을 따름.

개선 방향:
- 테이블에 Protocol, Sample Size, Follow-up 기간을 더 명확히 병기
- ComputationalMeasure, NeuralMeasure 빈 칸 보완 → task명 기반으로 RLDM construct와 연결해 최소 수준 자동 매핑 필요.
"""

(OUT / "03_response.md").write_text(response.text, encoding="utf-8")