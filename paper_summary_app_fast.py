import streamlit as st
import pandas as pd
import re
import os
import gc
import time
import tempfile
from io import BytesIO
from zipfile import ZipFile

# 무거운 라이브러리들은 필요할 때만 로드하도록 함수화
def load_fitz():
    """PyMuPDF 라이브러리를 필요한 시점에 로드"""
    import fitz
    return fitz

# 요약 모델 (필요할 때만 로드하는 지연 로딩 패턴 적용)
def get_summarizer():
    """요약 모델 로드 - 필요할 때만 메모리에 로드"""
    from transformers import pipeline
    # 가벼운 모델 선택 (BART-base는 Pegasus보다 가벼움)
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # CPU 사용

# 키워드 추출 모델 (필요할 때만 로드)
def get_keyword_model():
    """키워드 추출 모델 로드 - 필요할 때만 메모리에 로드"""
    from keybert import KeyBERT
    # 가벼운 임베딩 모델 사용 (기본 all-MiniLM-L6-v2)
    return KeyBERT()

# 메인 인터페이스 설정
st.set_page_config(page_title="📄 논문 요약기", layout="wide")
st.title("📄 논문 요약 도구 (최적화된 버전)")

# 배치 프로세싱을 위한 설정
st.sidebar.title("⚙️ 설정")
batch_size = st.sidebar.slider("배치 사이즈 (한 번에 처리할 PDF 수)", 1, 5, 3)
enable_summarization = st.sidebar.checkbox("텍스트 요약 활성화", True)
enable_keyword = st.sidebar.checkbox("키워드 추출 활성화", True)

# PDF에서 텍스트 추출 함수
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_content):
    """PDF에서 텍스트만 추출 (메모리 최적화)"""
    fitz = load_fitz()
    
    # 임시 파일로 저장하여 메모리 부담 줄이기
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_path = temp_file.name
    
    try:
        # 문서를 작은 단위로 처리
        with fitz.open(temp_path) as doc:
            # 첫 페이지 텍스트만 별도 추출 (메타데이터용)
            first_page_text = doc[0].get_text() if doc.page_count > 0 else ""
            
            # 전체 텍스트는 최대 20페이지까지만 처리 (메모리 절약)
            max_pages = min(20, doc.page_count)
            all_text = "\n".join([doc[i].get_text() for i in range(max_pages)])
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return all_text, first_page_text

# 결론 추출 함수
def extract_conclusion(text):
    """PDF에서 결론 부분만 추출"""
    text = text.lower()
    conclusion_text = ""
    
    # 결론 섹션 찾기
    for marker in ["conclusion", "conclusions", "summary"]:
        match = re.search(r"\b" + marker + r"\b", text)
        if match:
            start = match.end()
            end = len(text)
            
            # 다음 섹션 찾기
            for stop in ["reference", "acknowledgment", "bibliography"]:
                pos = text.find(stop, start)
                if pos != -1:
                    end = min(end, pos)
            
            conclusion_text = text[start:end].strip()
            break
    
    # 너무 긴 경우 최대 1000자로 제한 (메모리 절약)
    return conclusion_text[:1000] if conclusion_text else text[:1000]

# 제목과 저자 추출 함수 (간소화된 버전)
def extract_metadata(text):
    """제목과 저자 정보 추출 (간소화된 버전)"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 제목은 보통 처음 몇 줄 중 가장 긴 줄
    title_candidates = [line for line in lines[:10] if 3 < len(line.split()) < 20]
    title = max(title_candidates, key=len) if title_candidates else "제목 추출 실패"
    
    # 저자는 보통 이메일이나 소속이 포함된 줄 근처
    author = "저자 정보 없음"
    for i, line in enumerate(lines[:20]):
        if '@' in line or ('university' in line.lower() and i > 0):
            author = lines[i-1]
            break
    
    return title, author

# 배치 처리 함수
def process_pdfs_in_batches(files, batch_size=3):
    """PDF 파일들을 배치로 나누어 처리"""
    results = []
    
    # 배치 단위로 처리
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        batch_progress = st.progress(0)
        
        for j, file in enumerate(batch):
            # 진행상황 표시
            progress_text = f"처리 중: {file.name} ({j+1}/{len(batch)})"
            st.text(progress_text)
            batch_progress.progress((j+1)/len(batch))
            
            try:
                # PDF 파일 읽기
                pdf_bytes = file.read()
                file.seek(0)  # 파일 포인터 리셋
                
                # 텍스트 추출
                full_text, first_page = extract_text_from_pdf(pdf_bytes)
                
                # 메타데이터 추출
                title, author = extract_metadata(first_page)
                
                # 결론 추출
                conclusion = extract_conclusion(full_text)
                
                # 요약 및 키워드 추출 (선택적)
                summary = "요약 기능 비활성화됨"
                keywords = "키워드 기능 비활성화됨"
                
                if enable_summarization and conclusion:
                    # 요약 모델 로드 및 요약 생성
                    summarizer = get_summarizer()
                    summary = summarizer(conclusion[:500], max_length=80, min_length=30, 
                                       do_sample=False)[0]["summary_text"]
                    # 모델 제거
                    del summarizer
                    gc.collect()
                
                if enable_keyword and full_text:
                    # 키워드 모델 로드 및 키워드 추출
                    kw_model = get_keyword_model()
                    keywords_raw = kw_model.extract_keywords(full_text[:5000], 
                                                           keyphrase_ngram_range=(1, 2), 
                                                           stop_words='english', 
                                                           top_n=5)
                    keywords = ", ".join([k[0] for k in keywords_raw if len(k[0]) >= 3])
                    # 모델 제거
                    del kw_model
                    gc.collect()
                
                # 결과 저장
                results.append({
                    "파일명": file.name,
                    "제목": title,
                    "저자": author,
                    "결론 요약": summary,
                    "키워드": keywords
                })
                
                # 메모리 확보
                del full_text, first_page, conclusion
                gc.collect()
                
            except Exception as e:
                st.error(f"파일 처리 중 오류 발생: {file.name} - {str(e)}")
                # 오류가 발생해도 계속 처리
                results.append({
                    "파일명": file.name,
                    "제목": "오류 발생",
                    "저자": "오류 발생",
                    "결론 요약": f"처리 오류: {str(e)}",
                    "키워드": "오류"
                })
            
        # 배치 완료 후 메모리 정리
        batch_progress.empty()
        gc.collect()
    
    return results

# 메인 프로그램 UI
uploaded_files = st.file_uploader("📥 PDF 파일 업로드 (여러 개 선택 가능)", 
                                type="pdf", 
                                accept_multiple_files=True)

if uploaded_files:
    st.info(f"📂 {len(uploaded_files)}개 파일이 업로드되었습니다. '요약 시작' 버튼을 클릭하세요.")
    
    if st.button("🚀 요약 시작"):
        start_time = time.time()
        
        with st.spinner("논문 분석 중... 잠시만 기다려주세요"):
            # 배치 처리 실행
            results = process_pdfs_in_batches(uploaded_files, batch_size)
            
            # 데이터프레임 생성
            if results:
                df = pd.DataFrame(results)
                
                # 결과 표시
                st.success(f"✅ {len(results)}개 PDF 처리 완료! ({time.time() - start_time:.1f}초 소요)")
                st.dataframe(df)
                
                # 결과 다운로드 준비
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='논문요약')
                buffer.seek(0)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 Excel 파일로 다운로드",
                    data=buffer,
                    file_name="논문_요약_결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("처리된 결과가 없습니다.")
