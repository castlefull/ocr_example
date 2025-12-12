# 📁 paddle_ocr_extractor.py
from paddleocr import PaddleOCR
import json

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반)"""
    
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=True
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출"""
        result = self.ocr.ocr(image_path, cls=True)
        
        extracted_data = []
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            extracted_data.append({
                "bbox": bbox,
                "text": text,
                "confidence": confidence
            })
        
        return extracted_data
    
    def parse_quality_form(self, image_path):
        """품질검사서 필드별 파싱"""
        raw_data = self.extract_text(image_path)
        
        # 키워드 기반 필드 매핑
        field_keywords = {
            "lot_number": ["시료명", "시료번호", "Lot No"],
            "inspection_date": ["검사일자", "일자", "Date"],
            "inspection_equip": ["검사기기", "장비", "Equipment"],
            "temp":['온도'],
            "humidity":['습도'],
            "standard":['기준'],
            "method":['방법'],
            "spec":['규격'],
            "test_item":['항목'],
            "product_name": ["제품명", "품명", "Product"],
            "inspector": ["검사자", "담당자", "Inspector"],
            "result": ["판정", "결과", "Result", "합격", "불합격"],
            "record": ['검사기록','무게',"농도"]
        }
        
        parsed_result = {}
        
        for item in raw_data:
            text = item["text"]
            for field, keywords in field_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        # 다음 텍스트가 값일 가능성
                        parsed_result[field] = {
                            "value": text,
                            "confidence": item["confidence"]
                        }
        
        return parsed_result
    
    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)

# # 사용 예시
# if __name__ == "__main__":
#     ocr = QualityFormOCR(lang='korean')
#     result = ocr.parse_quality_form("quality_form_001.jpg")
#     ocr.to_json(result, "output_001.json")
#     print(result)
