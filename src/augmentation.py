# 📁 augmentation.py
import albumentations as A
import cv2
import os

class OCRAugmentor:
    """OCR 학습용 데이터 증강"""
    
    def __init__(self):
        self.transform = A.Compose([
            A.Rotate(limit=5, p=0.5),                    # 약간의 회전
            A.GaussNoise(var_limit=(10, 50), p=0.3),    # 가우시안 노이즈
            A.GaussianBlur(blur_limit=3, p=0.2),        # 블러
            A.RandomBrightnessContrast(p=0.3),          # 밝기/대비
            A.Perspective(scale=(0.02, 0.05), p=0.3),   # 원근 변환
            A.ImageCompression(quality_lower=70, p=0.2), # 압축 품질 저하
        ])
    
    def augment_dataset(self, input_dir, output_dir, augment_factor=10):
        """데이터셋 증강 (20장 → 200장)"""
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if not filename.endswith(('.jpg', '.png')):
                continue
                
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # 원본 저장
            cv2.imwrite(
                os.path.join(output_dir, filename), 
                image
            )
            
            # 증강 이미지 생성
            for i in range(augment_factor):
                augmented = self.transform(image=image)['image']
                aug_filename = f"{filename.split('.')[0]}_aug_{i}.jpg"
                cv2.imwrite(
                    os.path.join(output_dir, aug_filename), 
                    augmented
                )
        
        print(f"증강 완료: {len(os.listdir(output_dir))}장 생성")

# # 사용 예시
# if __name__ == "__main__":
#     augmentor = OCRAugmentor()
#     augmentor.augment_dataset(
#         input_dir="./raw_forms",
#         output_dir="./augmented_forms",
#         augment_factor=10
#     )
