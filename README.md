# 📄 논문 요약 도구 (Streamlit 기반)

최적화된 PDF 논문 요약 도구입니다. 업로드된 논문에서 텍스트를 추출하여 결론을 요약하고 주요 키워드를 제공합니다.

## 🧠 기능
- PDF 파일 다중 업로드 및 배치 처리
- 결론 섹션 자동 인식 및 요약
- 키워드 추출 (KeyBERT 기반)
- Excel 파일로 결과 다운로드

## 🚀 실행 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Streamlit 앱 실행:
```bash
streamlit run paper_summary_app_fast.py
```

## 📦 주요 패키지
- `streamlit`
- `transformers`
- `keybert`
- `PyMuPDF (fitz)`
- `openpyxl`
- `scikit-learn` (KeyBERT 사용 시 필요)

## 🧾 참고 모델
- 요약 모델: `sshleifer/distilbart-cnn-12-6`
- 임베딩 모델 (KeyBERT): `all-MiniLM-L6-v2`

## 📬 문의
개선 아이디어나 문의 사항은 GitHub Issue로 남겨주세요.
