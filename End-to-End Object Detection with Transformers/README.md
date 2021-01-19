# End-to-End Object Detection with Transformers - Facebook AI [[Paper Link](https://arxiv.org/abs/2005.12872)]
## Introduction
- 기존 객체 탐지 방법들은 너무 복잡하며 다양한 라이브러리를 활용함(NMS : non-maximum suppression 하나의 객체에 중복된 예측을 제거하는 작업)
- bounding box의 형태와 bounding box가 겹칠때 많은 문제 발생
- end-to-end 방식은 번역머신이나 음성인식에서 주로 사용되었지만 아직 object detection에서는 사용되고 있지 않았다. 이전의 연구들은 prior knowledge을 추가하거나 까다로운 형태와 경쟁하지 않는다고 한다. 다른 detector보다 새롭다

### 이분 매칭(biparite matching)을 통해 set prediction problem을 직접 해결 
- 이분 매칭을 통해 인스턴스가 중복되지 않도록 유도
- DETR의 구조 : CNN백본 모델을 통해 feature 맵을 생성하고 인코더-디코더 transformer를 거친다음 FFN(feed forward network)을 통해 결과값 출력
![제목 없음](https://user-images.githubusercontent.com/47103479/105000521-ca7e2100-5a71-11eb-95a0-ec6becd51cbb.png)

### Transformer
- transformer의 self-attention 메커니즘은 sequence의 모든 요소가 쌍으로 소통하며 중복된 prediction을 제거하는데 적합
- DETR은 모든 객체를 한번에 예측 , 병렬 디코딩을 통해 이분매칭
- RNN의 경우 autoregressive(자기회귀)에 포커스를 맞췄는데, DETR은 의 loss function은 predict할 때, predicted object의 순열이 변하지 않기 때문에 병렬적으로 작동
![제목 없음1](https://user-images.githubusercontent.com/47103479/105001464-25644800-5a73-11eb-9185-ed26f7ebed4f.png)

- Transformer
  * Attention을 통해 전체 이미지의 문맥 정보를 이해 
  * 이미지 내 각 인스턴스의 상호작용 파악 용이
  * 큰 bounding box에서의 거리가 먼 픽셀 간의 연관성 파악 용이
  
- Encoder
  * 이미지 특징(feature) 정보를 포함하고 있는 각 픽셀 위치 데이터를 입력받아 인코딩 수행
    * 인코더는 d * HW의 크기의 연속성을 띠는 feature map을 입력으로 받습니다(d : image feature, HW : 각각 픽셀 위치 정보)
    * self-attention map을 시각화 하면 개별 인스턴스를 적적히 분리한느것을 확인
    * 인코더의 self-attention 과정, 이렇게 인스턴스가 잘 나누어 진다면 디코더에서 object 위치와 클래스 예측하는건 매운 쉬운일 
    ![제목 없음2](https://user-images.githubusercontent.com/47103479/105001925-d8cd3c80-5a73-11eb-9c03-cc58d41ab18d.png)

 - Decoder
  * N 개의 object query를 초기 입력으로 받으며 인코딩된 정보를 활용
  * 각 object query는 이미지 내 서로 다른 고유한 인스턴스를 구별 
    * N개의 object Query(학습된 위치 임베딩)을 초기 입력으로 이용
    * 인코더가 global attention을 통해 인스턴스를 분리한 뒤에 디코더는 각 인스턴스의 클래스와 경계선을 추출
    * 디코더 attention은 상당히 지역적 다른 색상의 디코더의 prediction을 표시했는데 object의 head나 legs 쪽을 attention하는 것을 확인
    * 인코더가 global하게 attention하여 object의 인스턴스를 분리하는 반면 디코더는 object의 경계를 추출하기 위해 head와 legs에 attention한다
    ![제목 없음3](https://user-images.githubusercontent.com/47103479/105002210-3f525a80-5a74-11eb-9c1e-c4b2ce98d88b.png)

### DETR for panoptic segmentation
- DETR은 panoptic segmentation에 이용 될 수 있는데 원래 DETR의 구동 방식 대로, Box를 예측하고 mask head를 달아서 segmentation을 진행한다.

![제목 없음5](https://user-images.githubusercontent.com/47103479/105005420-a7a33b00-5a78-11eb-9e7e-0f9a89361e0a.png)

## Conclusion
- Object detection 분야에서 end-to-end 방식의 새로운 구조를 제안, DETR은 Faster R-CNN에 견주어 부족하지 않음
- DETR은 panoptic segmentation으로 확장이 쉽고, 경쟁력있는 결과
- Large object에서는 Faster R-CNN보다 좋은 성능

# Code Review[[Link](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Y6Jrz6xz71C0)]
## DETR
    class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

## model predictions

![제목 없음4](https://user-images.githubusercontent.com/47103479/105003138-a3295300-5a75-11eb-9e6e-59b8eafa15e6.png)


reference : DETR: End-to-End Object Detection with Transformers (꼼꼼한 딥러닝 논문 리뷰와 코드 실습) [[Link](https://www.youtube.com/watch?v=hCWUTvVrG7E&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=1)]
