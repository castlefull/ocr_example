# src/paddle_ocr_extractor.py (최종 수정 버전)

from paddleocr import PaddleOCR
import json
import os

class QualityFormOCR:
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,  # 3.x 파라미터
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
    
    def extract_text(self, image_path):
        # ✅ predict() 사용
        results = self.ocr.predict(image_path)
        
        extracted_data = []
        # ✅ res.json으로 안전하게 접근
        for res in results:
            data = res.json
            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])
            
            for text, score in zip(rec_texts, rec_scores):
                if text and str(text).strip():
                    extracted_data.append({
                        "text": str(text).strip(),
                        "confidence": float(score)
                    })
        
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

