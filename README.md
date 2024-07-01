# AI 기반 Recipe 추천 시스템

## 프로젝트 개요

이 프로젝트는 인공지능을 활용한 요리 레시피 추천 시스템을 개발하는 것을 목표로 합니다. 사용자가 업로드한 이미지를 분석하여 재료를 인식하고, 해당 재료를 활용한 레시피를 추천해주는 시스템입니다.

##도란도란 팀 

- **개발인원**: [팀원: 이준영], [팀원: 장우진], [팀원: 조준형], [팀원: 임태창], [팀원: 배은영], [팀원: 도규림]

## 프로젝트 개요

- **프로젝트명**: AI 기반 Recipe 추천 시스템
- **기간**: 2024년 5월31일 ~ 2024년 6월30일
- **목표**: YOLOv8을 이용한 객체 탐지 및 ViT를 이용한 이미지 분류를 통해 재료를 인식하고, 이를 바탕으로 레시피를 추천하는 시스템 개발

## 기술 스택

- **프레임워크**: TensorFlow, PyTorch
- **모델**: YOLOv8, ViT (Vision Transformer)
- **프로그래밍 언어**: Python, Flutter
- **기타 도구**: OpenCV, Flask

## 프로젝트 상세 내용

### 1. YOLOv8 Object Detection Test

YOLOv8 모델을 이용하여 업로드된 이미지에서 재료를 탐지합니다.

![yolo](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/4b2660c1-5e6f-47b1-8021-3d04227ed971)

### 2. Object Detection and BBox

탐지된 객체에 바운딩 박스를 표시하여 정확한 위치를 파악합니다.

![classfication](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/ffc93cf3-e260-4edd-a135-14ad6cd7c887)

### 3. ViT Classification

ViT 모델을 사용하여 탐지된 재료를 분류합니다.

![ViT_result](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/9bec3bbc-5f66-449e-8f91-2462d09a787f)
![ViT](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/7808ca0e-b432-4fc6-97bc-78cd313cd45d)

### 4. 실행 순서

시스템 실행 순서에 대한 다이어그램입니다.

![실행순서](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/a4017d0c-ad35-49cd-be68-03c581e97d91)

### 5. 앱 실행 영상

앱 실행 영상

#### 야채 2종류

[![야채 2종류](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/9c87b77a-946a-4c7a-81e9-85d3d31e7a33)](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/9c87b77a-946a-4c7a-81e9-85d3d31e7a33)

#### 야채 3종류

[![야채 3종류](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/748b3f04-382c-432b-9ba3-0bfd8e06598f)](https://github.com/JOJUNHYUNG0818/AI_vgt_recipe/assets/152590602/748b3f04-382c-432b-9ba3-0bfd8e06598f)

## 결론

이번 프로젝트는 인공지능 기술을 활용하여 일상생활에서 유용한 요리 레시피를 추천해주는 시스템을 개발하였습니다. 향후 다양한 재료와 레시피를 추가하여 시스템의 활용성을 높일 계획입니다.

## 참고 자료

- [YOLOv8 공식 문서](https://github.com/ultralytics/yolov5)
- [ViT 논문](https://arxiv.org/abs/2010.11929)

