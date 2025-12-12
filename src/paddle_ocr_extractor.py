from paddleocr import PaddleOCR
import json

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반)"""
    
    def __init__(
        self,
        lang: str = 'korean',
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
    ):
        # PaddleOCR 3.x: predict 파이프라인 + 탐지 파라미터 조정[web:169][web:199]
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            text_det_thresh=det_db_thresh,        # 예전 det_db_thresh 역할[web:201][web:196]
            text_det_box_thresh=det_db_box_thresh # 예전 det_db_box_thresh 역할
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 - predict() 기반"""
        try:
            results = self.ocr.predict(image_path)
        except Exception as e:
            print(f"OCR 실행 오류: {e}")
            return []
        
        extracted_data = []

        for res in results:
            try:
                data = res.json  # 3.x 표준 출력[web:169]
                rec_texts = data.get("rec_texts", [])
                rec_scores = data.get("rec_scores", [])
                rec_boxes = data.get("rec_boxes", [])

                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    if not text or not str(text).strip():
                        continue
                    extracted_data.append({
                        "bbox": box,
                        "text": str(text).strip(),
                        "confidence": float(score)
                    })
            except Exception as e:
                print(f"결과 파싱 오류: {e}, data: {data}")
                continue
        
        if not extracted_data:
            print("⚠ OCR 결과에서 텍스트를 찾지 못했습니다.")
        
        return extracted_data
    
    def parse_quality_form(self, image_path):
        """품질검사서 필드별 파싱"""
        raw_data = self.extract_text(image_path)
        
        if not raw_data:
            return {"error": "텍스트 추출 실패", "full_text": ""}
        
        field_keywords = {
            "lot_number": ["시료명", "시료번호", "Lot No", "LOT"],
            "inspection_date": ["검사일자", "일자", "Date"],
            "inspection_equip": ["검사기기", "장비", "Equipment"],
            "temp": ['온도', 'Temperature'],
            "humidity": ['습도', 'Humidity'],
            "standard": ['기준', 'Standard'],
            "method": ['방법', 'Method'],
            "spec": ['규격', 'Specification'],
            "test_item": ['항목', 'Item'],
            "product_name": ["제품명", "품명", "Product"],
            "inspector": ["검사자", "담당자", "Inspector"],
            "result": ["판정", "결과", "Result", "합격", "불합격"],
            "record": ['검사기록', '무게', "농도"]
        }
        
        parsed_result = {}
        full_text = " ".join([item["text"] for item in raw_data])
        
        for item in raw_data:
            text = item["text"]
            for field, keywords in field_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        if field not in parsed_result or \
                           item["confidence"] > parsed_result[field]["confidence"]:
                            parsed_result[field] = {
                                "value": text,
                                "confidence": item["confidence"]
                            }
        
        parsed_result["full_text"] = full_text
        return parsed_result
    
    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
