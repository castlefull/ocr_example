# src/paddle_ocr_extractor.py

from paddleocr import PaddleOCR
import json
import os

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반) - 수정된 버전"""
    
    def __init__(
        self,
        lang: str = 'korean',
        det_db_thresh: float = 0.3,     # Streamlit 슬라이더와 연동됨
        det_db_box_thresh: float = 0.5, # Streamlit 슬라이더와 연동됨
    ):
        # 1. use_angle_cls=True: 회전된 문서 인식율 향상
        # 2. show_log=False: 불필요한 콘솔 로그 제거
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True, 
            show_log=False,
            det_db_thresh=det_db_thresh,        
            det_db_box_thresh=det_db_box_thresh
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 - 표준 ocr() 메서드 사용"""
        
        if not os.path.exists(image_path):
            print(f"❌ 파일 없음: {image_path}")
            return []

        try:
            # 기존 predict() -> ocr()로 변경 (가장 중요한 수정 사항)
            results = self.ocr.ocr(image_path, cls=True)
        except Exception as e:
            print(f"❌ OCR 실행 중 오류 발생: {e}")
            return []
        
        # 결과가 없는 경우 처리
        if results is None or len(results) == 0 or results[0] is None:
            return []
        
        extracted_data = []

        # PaddleOCR 결과 파싱 (리스트 구조 분해)
        for line in results[0]:
            try:
                # line 구조: [[x값들], ('텍스트', 점수)]
                box = line[0]
                text_info = line[1]
                text = text_info[0]
                score = text_info[1]

                if text and str(text).strip():
                    extracted_data.append({
                        "bbox": box,
                        "text": str(text).strip(),
                        "confidence": float(score)
                    })
            except Exception:
                continue
        
        return extracted_data

    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        try:
            # 디렉토리가 없으면 생성 (FileNotFoundError 방지)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"JSON 저장 실패: {e}")
            return False

    # (필요하다면 parse_quality_form 등의 추가 메서드도 여기에 포함)
