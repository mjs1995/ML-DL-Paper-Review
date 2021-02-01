# Rich feature hierarchies for accurate object detection and semantic segmentation - Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
## R-CNN [[Paper Link](https://arxiv.org/abs/1311.2524)]
## 들어가기에 앞서
- 컴퓨터비전의 4가지 분류
  * 1. Classification
  * 2. Object Detection
  * 3. Image Segmentation
  * 4. Visual relationship

 ![제목 없음](https://user-images.githubusercontent.com/47103479/106444224-cfef5880-64c0-11eb-9a5e-248c4b1e16b2.png)

  * Classification : Single object에 대해서 object의 클래스를 분류하는 문제
  * Classification + Localization : Single object에 대해서 object의 위치를 bounding box로 찾고 (Localization) + 클래스를 분류하는 문제
  * Object Detection : Multiple objects에서 각각의 object에 대해 Classification + Localization을 수행
    * 1- stage detector
      ![제목 없음2](https://user-images.githubusercontent.com/47103479/106445310-332dba80-64c2-11eb-8215-f4c1efb39201.png)
      * ROI 영역을 먼저 추출하지 않고 전체 Image에 대해서 convolution network로 classification, box regression(localization)을 수행
      * 여러 노이즈(object가 섞여있는 전체 image에서) 수행하는 정확도 떨어지는 대신에 간단하고 쉽고 속도가 빠르다
    * 2- stage detector
      ![제목 없음1](https://user-images.githubusercontent.com/47103479/106445149-0083c200-64c2-11eb-8211-f0e1b1e82d66.png)
      * Selective search, Region proposal network와 같은 알고리즘을 및 네트워크를 통해 object가 있을만한 영역을 우선 뽑아낸다. 이 영역을 RoI(Region of Interest)라고 한다. 각 영역들을 convolution network를 통해 classification, box regression(localization)을 수행 
    
  * Instance Segmentation : object의 위치를 bounding box가 아닌 실제 edge로 찾는 것이다.

## Abstract
  * 객체를 localize 및 segment하기 위해 bottom-up방식의 region proposal(지역 제안)에 Convolutional Neural Network를 적용
  * domain-specific fine-tuning을 통한 supervised pre-training을 적용
  * 저자는 R-CNN(Regions with CNN features)이라고 명시 - CNN과 Region proposal이 결합되었기 때문
  
## Introduction
  * 지난 10년간 SIFT와 HOG(gradient 기반의 특징점 추출 알고리즘)가 가장 많이 사용되었는데 back-propagation이 가능한 SGD(Stochastic Gradient Descent)기반의 CNN(Convolutional Neural Networks)이 등장하면서 PASCAL VOC object detection 굉장한 성능을 보임 
  * R-CNN 프로세스
  
    ![제목 없음3](https://user-images.githubusercontent.com/47103479/106445739-bfd87880-64c2-11eb-838b-6f2cc4302509.png)
    * Input 이미지로부터 2,000개의 독립적인 region proposal을 생성
    * CNN을 통해 각 proposal 마다 고정된 길이의 feature vector를 추출(CNN 적용 시 서로 다른 region shape에 영향을 받지 않기 위해 fixed-size로 이미지를 변경)
    * 이후, 각 region 마다 category-specific linear SVM을 적용하여 classification을 수행

## Object detection with R-CNN
  * category-independent한 region proposals를 생성
  * 각 region으로부터 feature vector를 추출하기 위한 large CNN
  * classification을 위한 linear SVMs
  
  * Region proposals
  

  
    * 카테고리 독립적인 region proposal을 생성하기 위한 방법은 여러가지가 있는데 해당 논문에서는 이전 detection 작업들과 비교하기 위하여 Selective Search라는 최적의 region proposal를 제안하는 기법을 사용하여 독립적인 region proposal을 추출
      * Selective Search
        * 1.이미지의 초기 세그먼트를 정하여, 수많은 region 영역을 생성
        * 2.greedy 알고리즘을 이용하여 각 region을 기준으로 주변의 유사한 영역을 결합
        * 3.결합되어 커진 region을 최종 region proposal로 제안
  * Feature extraction
  
    ![제목 없음4](https://user-images.githubusercontent.com/47103479/106446068-29f11d80-64c3-11eb-935f-a0ddeff7d430.png)
    * Selective Search를 통해 region proposal로부터 cnn을 사용하여 4096차원의 feature vector를 추출
    * 5개의 convolutional layer 와 2개의 fully connected layer로 전파 
    * 각 region은 227 * 227 RGB의 고정된 사이즈로 변환
    
  * Trainging
    * 학습에 사용된 CNN 모델의 경우 ILSVRC 2012 데이터 셋으로 미리 학습된 pre-trained CNN(AlexNet)모델
    
  * Domain-specific fine-tuning
    * IoU는 Area of Overlap(교집합) / Area of Union(합집합)으로 계산
    * NMS(Non-maximum suppresion)
      * 1.예측한 bounding box들의 예측 점수를 내림차순으로 정렬
      * 2.높은 점수의 박스부터 시작하여 나머지 박스들 간의 IoU를 계산
      * 3.IoU값이 지정한 threhold보다 높은 박스를 제거
      * 4.최적의 박스만 남을 떄까지 위 과정을 반복
      
      ![제목 없음5](https://user-images.githubusercontent.com/47103479/106458133-45fcbb00-64d3-11eb-888c-4923701d15ad.png)
      
## Problems
  - R-CNN은 selective search 알고리즘를 통한 region proposal 작업 그리고 NMS 알고리즘 작업 등은 CPU 연산에 의해 이루어 지기 때문에 굉장히 많은 연산량 및 시간이 소모
  - 이러한 문제점을 개선해서 나온것이 Fast R-CNN과 Faster R-CNN      
  -  Fast R-CNN
  
   ![제목 없음6](https://user-images.githubusercontent.com/47103479/106458309-8bb98380-64d3-11eb-9d35-3bbcfa6de230.png)
    
   * Fast R-CNN은 먼저 전체 이미지가 ConvNet의 input으로 입력
   * 이미지는 ConvNet을 통과하며 feature map을 추출하게 되고, 이 feature map은 selectice search 기반의 region proposal을 통해 RoI(Regions of Interest)를 뽑아낸다.
   * Fast RCNN 또한 많은 연산을 필요로 하는 Selective Search 기법이 작동을 하므로 큰 데이터 셋에 적용하는데는 한계
    
  - Faster R-CNN
  
    ![제목 없음7](https://user-images.githubusercontent.com/47103479/106458459-c4f1f380-64d3-11eb-81dc-b37ea51fe5ca.png)
    
    * Faster R-CNN에서는 selective search 알고리즘을 없애고 Region Proposal Networks(RPN)라는 뉴럴 네트워크를 추가하여 region proposal을 예측
    * 예측된 region proposal은 Fast R-CNN과 유사하게 RoI Pooling layer를 거치며 모든 region을 같은 크기로 고정 후, Classification 및 Bounding Box Regreesion이 수행




  
