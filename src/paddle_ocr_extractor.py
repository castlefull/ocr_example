from paddleocr import PaddleOCR
import json

class QualityFormOCR:
    """품질검사서 OCR 추출기 (PaddleOCR 기반)"""
    
    def __init__(self, lang='korean'):
        # PaddleOCR 3.0+ 최소 파라미터
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang
        )
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 - 안전한 파싱"""
        try:
            result = self.ocr.ocr(image_path)
        except Exception as e:
            print(f"OCR 실행 오류: {e}")
            return []
        
        extracted_data = []
        
        # 결과가 None이거나 비어있는지 확인
        if not result:
            print("OCR 결과가 None입니다.")
            return extracted_data
        
        if not result[0]:
            print("OCR 결과가 비어있습니다.")
            return extracted_data
        
        # 각 라인 안전하게 파싱
        for idx, line in enumerate(result[0]):
            try:
                # 기본 구조: [bbox, (text, confidence)]
                if not line or len(line) < 2:
                    print(f"라인 {idx}: 구조 불완전 - {line}")
                    continue
                
                bbox = line[0]
                text_info = line[1]
                
                # text와 confidence 안전하게 추출
                if isinstance(text_info, (list, tuple)):
                    if len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    elif len(text_info) == 1:
                        text = str(text_info[0])
                        confidence = 1.0
                    else:
                        print(f"라인 {idx}: text_info 비어있음 - {text_info}")
                        continue
                elif isinstance(text_info, str):
                    text = text_info
                    confidence = 1.0
                else:
                    print(f"라인 {idx}: 알 수 없는 형식 - {text_info}")
                    continue
                
                # 빈 텍스트 스킵
                if not text or not text.strip():
                    continue
                
                extracted_data.append({
                    "bbox": bbox,
                    "text": text.strip(),
                    "confidence": confidence
                })
                
            except Exception as e:
                print(f"라인 {idx} 파싱 오류: {e}, 데이터: {line}")
                continue
        
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
