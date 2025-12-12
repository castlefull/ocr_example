from paddleocr import PaddleOCR
import json
import os

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 3.x)"""
    
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출"""
        results = self.ocr.predict(image_path)
        
        extracted_data = []
        
        for res in results:
            # ✅ 수정: res.json에서 직접 접근
            data = res.json
            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])
            rec_boxes = data.get("rec_boxes", [])
            
            # ✅ 텍스트와 점수 매칭
            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                if text and str(text).strip():
                    extracted_data.append({
                        "text": str(text).strip(),
                        "confidence": float(score),
                        "bbox": box
                    })
        
        return extracted_data
    
    def save_results(self, image_path, output_dir="output"):
        """공식 예제처럼 이미지/JSON 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.ocr.predict(image_path)
        
        for res in results:
            res.save_to_img(output_dir)
            res.save_to_json(output_dir)
        
        return results
    
    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"JSON 저장 실패: {e}")
            return False
