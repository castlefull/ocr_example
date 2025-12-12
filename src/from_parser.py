# 📁 form_parser.py
import json
import csv
import re
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class QualityInspectionRecord:
    """품질검사 기록 데이터 클래스"""
    lot_number: str
    inspection_date: str
    product_name: str
    inspector_name: str
    
    # 측정값들
    measurement_1: Optional[float] = None
    measurement_2: Optional[float] = None
    measurement_3: Optional[float] = None
    
    # 판정
    result: str = ""  # 합격/불합격
    remarks: str = ""

class FormToStructuredData:
    """OCR 결과를 정형 데이터로 변환"""
    
    def __init__(self):
        self.patterns = {
            "lot_number": r"(?:LOT|로트)[:\s]*([A-Z0-9\-]+)",
            "date": r"(\d{4}[-/년]\d{1,2}[-/월]\d{1,2}일?)",
            "measurement": r"(\d+\.?\d*)\s*(mm|kg|%|℃)?",
        }
    
    def extract_field(self, text, pattern):
        """정규식으로 필드 추출"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def parse_ocr_result(self, ocr_data: list) -> QualityInspectionRecord:
        """OCR 데이터를 구조화된 레코드로 변환"""
        
        full_text = " ".join([item["text"] for item in ocr_data])
        
        record = QualityInspectionRecord(
            lot_number=self.extract_field(
                full_text, self.patterns["lot_number"]
            ) or "",
            inspection_date=self.extract_field(
                full_text, self.patterns["date"]
            ) or "",
            product_name="",
            inspector_name="",
        )
        
        # 판정 결과 추출
        if "합격" in full_text:
            record.result = "합격"
        elif "불합격" in full_text:
            record.result = "불합격"
        
        return record
    
    def to_csv(self, records: list, output_path: str):
        """레코드 리스트를 CSV로 저장"""
        if not records:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(records[0]).keys())
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
    
    def to_json(self, records: list, output_path: str):
        """레코드 리스트를 JSON으로 저장"""
        data = [asdict(record) for record in records]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# # 사용 예시
# if __name__ == "__main__":
#     from paddle_ocr_extractor import QualityFormOCR
    
#     ocr = QualityFormOCR()
#     parser = FormToStructuredData()
    
#     # 여러 검사지 처리
#     records = []
#     for i in range(1, 21):
#         ocr_result = ocr.extract_text(f"quality_form_{i:03d}.jpg")
#         record = parser.parse_ocr_result(ocr_result)
#         records.append(record)
    
#     # 저장
#     parser.to_csv(records, "quality_inspection_data.csv")
#     parser.to_json(records, "quality_inspection_data.json")
