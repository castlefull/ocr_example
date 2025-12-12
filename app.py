import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import os

# 기존 src 폴더의 클래스들을 import
from src.preprocess import ImagePreprocessor
from src.paddle_ocr_extractor import QualityFormOCR
from src.form_parser import FormToStructuredData

# 페이지 설정
st.set_page_config(page_title="품질검사서 OCR 데모", layout="wide")

# 제목
st.title("📋 손글씨 품질검사서 OCR 시스템")
st.markdown("---")

# 사이드바 - 실행 단계 선택
st.sidebar.title("실행 단계 선택")
step = st.sidebar.radio(
    "처리 단계:",
    ["1️⃣ 이미지 전처리", "2️⃣ OCR 텍스트 추출", "3️⃣ 정형 데이터 변환", "🔄 전체 파이프라인"]
)

# 파일 업로드
uploaded_file = st.file_uploader(
    "품질검사서 이미지를 업로드하세요", 
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file is not None:
    # 이미지 읽기
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 이미지 표시
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_column_width=True)
    
    # 임시 파일 저장
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 단계별 처리
    if step == "1️⃣ 이미지 전처리":
        st.subheader("Step 1: 이미지 전처리")
        
        if st.button("전처리 실행"):
            with st.spinner("전처리 중..."):
                # 기존 preprocess.py의 클래스 사용
                preprocessor = ImagePreprocessor()
                processed = preprocessor.preprocess_pipeline(temp_path)
                
                with col2:
                    st.subheader("전처리된 이미지")
                    st.image(processed, use_column_width=True, channels="GRAY")
                
                # 저장 옵션
                processed_path = f"data/processed/{uploaded_file.name}"
                cv2.imwrite(processed_path, processed)
                st.success(f"✅ 전처리 완료! 저장 경로: {processed_path}")
    
    elif step == "2️⃣ OCR 텍스트 추출":
        st.subheader("Step 2: OCR 텍스트 추출")
        
        if st.button("OCR 실행"):
            with st.spinner("텍스트 추출 중... (약 10초 소요)"):
                # 기존 paddle_ocr_extractor.py의 클래스 사용
                ocr = QualityFormOCR(lang='korean')
                extracted_data = ocr.extract_text(temp_path)
                
                if not extracted_data:
                    st.warning("⚠️ 텍스트를 찾을 수 없습니다.")
                    st.info("""
                    **가능한 원인:**
                    - 이미지가 너무 작거나 해상도가 낮음
                    - 텍스트가 흐리거나 배경과 구분이 안 됨
                    - 손글씨가 너무 흘림체
                    
                    **해결 방법:**
                    - 300 DPI 이상의 선명한 이미지 사용
                    - 조명이 좋은 환경에서 촬영
                    - 텍스트가 잘 보이는 영역만 크롭
                    """)
                else:
                    st.success(f"✅ {len(extracted_data)}개 텍스트 발견!")
                    for item in extracted_data:
                        st.write(f"- {item['text']} ({item['confidence']:.2%})")
                # 결과 표시
                st.subheader("추출된 텍스트")
                for idx, item in enumerate(extracted_data):
                    with st.expander(f"텍스트 {idx+1}: {item['text']}", expanded=True):
                        st.write(f"**신뢰도:** {item['confidence']:.3f}")
                        st.write(f"**좌표:** {item['bbox']}")
                
                # JSON 저장 및 다운로드
                output_path = f"output/ocr_{uploaded_file.name}.json"
                ocr.to_json(extracted_data, output_path)
                
                with open(output_path, 'r', encoding='utf-8') as f:
                    json_str = f.read()
                
                st.download_button(
                    label="📥 JSON 다운로드",
                    data=json_str,
                    file_name=f"ocr_result.json",
                    mime="application/json"
                )
    
    elif step == "3️⃣ 정형 데이터 변환":
        st.subheader("Step 3: 정형 데이터 변환")
        
        if st.button("정형화 실행"):
            with st.spinner("데이터 변환 중..."):
                # OCR 실행 (전처리 포함)
                ocr = QualityFormOCR(lang='korean')
                ocr_result = ocr.extract_text(temp_path)
                
                # 기존 form_parser.py의 클래스 사용
                parser = FormToStructuredData()
                record = parser.parse_ocr_result(ocr_result)
                
                # 결과를 DataFrame으로 표시
                df = pd.DataFrame([record.__dict__])
                st.subheader("추출된 정형 데이터")
                st.dataframe(df, use_container_width=True)
                
                # CSV 저장 및 다운로드
                csv_path = "output/quality_inspection_data.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                csv_str = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_str,
                    file_name="quality_inspection_data.csv",
                    mime="text/csv"
                )
    
    elif step == "🔄 전체 파이프라인":
        st.subheader("전체 파이프라인 실행")
        
        if st.button("전체 프로세스 실행"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. 전처리
            status_text.text("⚙️ 1/3: 이미지 전처리 중...")
            preprocessor = ImagePreprocessor()
            processed = preprocessor.preprocess_pipeline(temp_path)
            processed_path = f"temp_processed_{uploaded_file.name}"
            cv2.imwrite(processed_path, processed)
            progress_bar.progress(33)
            
            # 2. OCR
            status_text.text("📄 2/3: OCR 텍스트 추출 중...")
            ocr = QualityFormOCR(lang='korean')
            ocr_result = ocr.extract_text(processed_path)
            progress_bar.progress(66)
            
            # 3. 정형화
            status_text.text("🔄 3/3: 정형 데이터 변환 중...")
            parser = FormToStructuredData()
            record = parser.parse_ocr_result(ocr_result)
            df = pd.DataFrame([record.__dict__])
            progress_bar.progress(100)
            status_text.text("✅ 완료!")
            
            # 결과 표시
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("전처리된 이미지")
                st.image(processed, channels="GRAY")
            
            with col2:
                st.subheader("추출된 정형 데이터")
                st.dataframe(df, use_container_width=True)
            
            # 다운로드 버튼
            csv_str = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 결과 CSV 다운로드",
                data=csv_str,
                file_name="final_result.csv",
                mime="text/csv"
            )
            
            # 임시 파일 삭제
            os.remove(processed_path)
    
    # 임시 파일 정리
    os.remove(temp_path)

else:
    st.info("👆 왼쪽 사이드바에서 처리 단계를 선택한 후 이미지를 업로드하세요.")

