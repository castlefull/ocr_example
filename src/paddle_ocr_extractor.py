# src/paddle_ocr_extractor.py (수정 버전)

from paddleocr import PaddleOCR
import json
import os

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반) - 임계값 인자 수정 반영"""
    
    def __init__(
        self,
        lang: str = 'korean',
        # det_db_thresh와 det_db_box_thresh는 __init__에서 제거함!
        # 대신, 인스턴스 변수로 저장하여 extract_text에서 사용합니다.
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
    ):
        # 1. PaddleOCR 객체 초기화 시, 라이브러리가 모르는 인자를 전달하지 않도록 합니다.
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True, 
            show_log=False,
            # det_db_thresh, det_db_box_thresh 인자는 이제 여기에 넣지 않습니다!
        )
        
        # 임계값은 저장해 둡니다.
        self._det_db_thresh = det_db_thresh
        self._det_db_box_thresh = det_db_box_thresh
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 - 표준 ocr() 메서드 사용 및 임계값 전달"""
        
        if not os.path.exists(image_path):
            print(f"❌ 파일 없음: {image_path}")
            return []
        
        # 2. ocr() 호출 시, det_params 인자를 통해 임계값을 전달합니다.
        # 이것이 신버전 PaddleOCR에서 권장되는 설정 방식입니다.
        det_params = {
            'det_db_thresh': self._det_db_thresh,
            'det_db_box_thresh': self._det_db_box_thresh
        }

        try:
            results = self.ocr.ocr(image_path, cls=True, det_params=det_params)
        except Exception as e:
            print(f"❌ OCR 실행 중 오류 발생: {e}")
            return []
        
        # ... (이하 결과 파싱 로직은 동일) ...
        if results is None or len(results) == 0 or results[0] is None:
            return []
        
        extracted_data = []

        for line in results[0]:
            try:
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

    # ... (to_json 메서드는 그대로 유지) ...
    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"JSON 저장 실패: {e}")
            return False
