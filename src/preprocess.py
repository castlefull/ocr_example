# 📁 preprocess.py
import cv2
import numpy as np

class ImagePreprocessor:
    """품질검사서 이미지 전처리 클래스"""
    
    def __init__(self):
        pass
    
    def grayscale(self, image):
        """그레이스케일 변환"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def remove_noise(self, image):
        """노이즈 제거 - 미디안 블러"""
        return cv2.medianBlur(image, 3)
    
    def binarize(self, image):
        """이진화 - Otsu 방식"""
        _, binary = cv2.threshold(
            image, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary
    
    # def deskew(self, image):
    #     """기울기 보정"""
    #     coords = np.column_stack(np.where(image > 0))
    #     angle = cv2.minAreaRect(coords)[-1]
        
    #     if angle < -45:
    #         angle = -(90 + angle)
    #     else:
    #         angle = -angle
            
    #     (h, w) = image.shape[:2]
    #     center = (w // 2, h // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     rotated = cv2.warpAffine(
    #         image, M, (w, h),
    #         flags=cv2.INTER_CUBIC,
    #         borderMode=cv2.BORDER_REPLICATE
    #     )
    #     return rotated
    
    def preprocess_pipeline(self, image_path):
        """전체 전처리 파이프라인"""
        image = cv2.imread(image_path)
        gray = self.grayscale(image)
        denoised = self.remove_noise(gray)
        binary = self.binarize(denoised)
        deskewed = self.deskew(binary)
        return deskewed

# # 사용 예시
# if __name__ == "__main__":
#     preprocessor = ImagePreprocessor()
#     processed = preprocessor.preprocess_pipeline("quality_form_001.jpg")
#     cv2.imwrite("processed_001.jpg", processed)

