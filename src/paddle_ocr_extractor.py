# src/paddle_ocr_extractor.py (최종 수정 버전)

from paddleocr import PaddleOCR
import json
import os

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반) - 극단적인 파라미터 간소화 적용"""
    
    def __init__(
        self,
        lang: str = 'korean',
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
    ):
        # PaddleOCR 초기화: 'lang' 외의 모든 옵션(use_angle_cls, show_log 등)을 제거하고
        # 라이브러리 기본값에 의존하도록 간소화합니다.
        try:
            self.ocr = PaddleOCR(lang=lang)
        except ValueError as e:
            # lang='korean'이 문제가 된다면, 일단 기본값으로 재시도
            if "Unknown argument" in str(e):
                print("경고: 언어 파라미터가 오류를 일으킵니다. 기본값으로 재시도합니다.")
                self.ocr = PaddleOCR() 
            else:
                raise e

        # 임계값은 인스턴스 변수로 저장하여 ocr() 호출 시 사용합니다.
        self._det_db_thresh = det_db_thresh
        self._det_db_box_thresh = det_db_box_thresh
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 - det_params를 통해 임계값 전달"""
        
        if not os.path.exists(image_path):
            print(f"❌ 파일 없음: {image_path}")
            return []
        
        # 텍스트 탐지 임계값 설정
        det_params = {
            'det_db_thresh': self._det_db_thresh,
            'det_db_box_thresh': self._det_db_box_thresh
        }

        try:
            # cls=True (회전 보정)와 det_params (임계값)를 ocr() 메서드 호출 시 전달합니다.
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
