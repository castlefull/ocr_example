# 📁 paddle_ocr_extractor.py
from paddleocr import PaddleOCR
import json

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반)"""
    
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False  # 로그 숨기기
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
                    # 예상치 못한 형식
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
                print(f"라인 파싱 오류: {e}, 데이터: {line}")
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
                        # 이미 해당 필드가 없거나, 더 높은 신뢰도면 업데이트
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
    
    def to_dict(self, parsed_data):
        """딕셔너리 반환 (Streamlit용)"""
        return parsed_data


# 사용 예시
if __name__ == "__main__":
    ocr = QualityFormOCR(lang='korean')
    
    # 텍스트 추출만
    extracted = ocr.extract_text("quality_form_001.jpg")
    print("추출된 텍스트:")
    for item in extracted:
        print(f"- {item['text']} (신뢰도: {item['confidence']:.2f})")
    
    # 필드별 파싱
    result = ocr.parse_quality_form("quality_form_001.jpg")
    ocr.to_json(result, "output_001.json")
    print("\n파싱 결과:")
    print(result)
