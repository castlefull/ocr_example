# 📁 paddle_ocr_extractor.py
from paddleocr import PaddleOCR
import json

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반)"""
    
    def __init__(self, lang='korean'):
        # PaddleOCR 3.0+ 호환: 최소 파라미터만 사용
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang
            # show_log, use_gpu 등 제거됨!
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출"""
        result = self.ocr.ocr(image_path)
        
        extracted_data = []
        
        # 결과가 비어있는지 확인
        if not result or not result[0]:
            return extracted_data
        
        for line in result[0]:
            try:
                bbox = line[0]
                
                # PaddleOCR 3.0+ 호환: 안전하게 text와 confidence 추출
                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                    text = str(line[1][0])
                    confidence = float(line[1][1])
                elif isinstance(line[1], dict):
                    text = str(line[1].get('text', ''))
                    confidence = float(line[1].get('confidence', 1.0))
                else:
                    text = str(line[1])
                    confidence = 1.0
                
                # 빈 텍스트 스킵
                if not text.strip():
                    continue
                
                extracted_data.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence
                })
                
            except (IndexError, TypeError, ValueError) as e:
                print(f"라인 파싱 오류: {e}")
                continue
        
        return extracted_data
    
    def parse_quality_form(self, image_path):
        """품질검사서 필드별 파싱"""
        raw_data = self.extract_text(image_path)
        
        # 키워드 기반 필드 매핑
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
            "record": ['검사기록', '무게', "농도", "Weight", "Concentration"]
        }
        
        parsed_result = {}
        full_text = " ".join([item["text"] for item in raw_data])
        
        # 각 필드별로 매칭 시도
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
        
        # 전체 텍스트도 포함
        parsed_result["full_text"] = full_text
        
        return parsed_result
    
    def to_json(self, parsed_data, output_path):
        """JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
