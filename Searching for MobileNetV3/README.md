# Searching for MobileNetV3 - Google AI, Google Brain [[Paper Link](https://arxiv.org/abs/1905.02244)]
## Abstract
- MobileNetV3는 NetAdap 알고리즘으로 보완된 하드웨어-인식(Hardware-aware) 네트워크 아키테쳐 탐색(Network Architecture Search, NAS)의 결합을 통해 모바일 폰의 CPUs에 맞춰 튜닝
- MobileNet v3는 mobile phone CPU에 최적화
- 자동화된 탐색 알고리즘과 네트워크 설계가 기술의 전반적인 상태를 개선시키는 보완적 방법을 활용하여 어떻게 상호작용하는지 탐색
- MobileNetV3-Large는 MobileNetV2에 비해 ImageNet classification에서 3.2% 정확하면서도 20%의 latency가 개선됨

## Introduction
- 효율적인 on-device 인공신경망은 mobile 적용 시대에 있어서 매우 흔함
  * 이러한 on-device 딥러닝은 사용자의 개인정보를 서버로 전송하지 않고도 사용자에 최적화된 구동을 위해 필수로 필요한 분야
- 논문의 목표는 accuracy-latency 최적화를 통해 mobile환경에서 최고의 mobile computer vision architecture를 제안하는 것
  * Complementary search techniques
  * Mobile setting에 효율적인 새로운 non-linearities practical version을 제안
  * 새로운 효율적인 네트워크 디자인
  * 새로운 효율적인 segmentation decoder
-  mobile phone에서 다양하고 광범위한 방법으로 효율성등을 실험적으로 검증
  * Section 2에선 related work에 대해서 다룸
  * Section 3에선 mobile model들의 efficient building block들에서 사용된 방법들을 리뷰
  * Section 4에선 NAS와 MnasNet, NetAdapt 알고리즘들의 상호적인 보완적 특성을 다룸
  * Section 5에선 joint search를 통해 찾아진 모델의 효율을 높히는 새로운 architecture design을 설명
  * Section 6에선 classification, detection, segmentation task를 이용해 모델의 효율과 각 적용요소들의 contribution에 대해 실험하고 결과를 설명
  * Section 7에선 결론 및 future work를 다룸

## Related Work
- 최근 다방면에서 뉴럴넷의 최적의 정확도-효율 trade-off를 찾기위한 다양한 연구들이 수행
- SqueezeNet[22]은 squeeze와 expand 모듈과 1x1 컨벌루션을 광범위하게 사용해 파라미터 수를 줄이는것에 중점을 두고 연구되었음
- MobileNetV1[19]은 연산 효율 증가를 위해 depthwise separable convolution을 사용함
- MobileNetV2[39]은 위의 방법을 이용하면서도 resource-efficient한 inverted residual block과 linear bottleneck을 제안함
- ShuffleNet[49]은 group convolution과 channel shuffle 연산을 활용해 연산량을 줄임
- CondenseNet[21]은 모델 학습단에서 group convolution을 학습시켜 feature 재사용을 위한 layer간 dense connection을 활용했음
- ShiftNet[46]은 연산비용이 비싼 spatial convolution을 대체하기 위해 point-wise convolution을 중간에 끼워넣은 shift operation을 제안함


## Efficient Mobile Building Blocks
- Mobile model들은 엄청 효율적인 building block들을 이용해 만들어짐
- MobileNetV1[19]은 depth-wise separable convolution을 이용해 일반적인 conv를 대체하는 방법을 제안함
- Depthwise separable convolution은 효과적으로 일반 conv를 factorize했으며, 이는 feature 생성에서 spatial filtering을 분리시킨 결과

![제목 없음](https://user-images.githubusercontent.com/47103479/105128934-5e5df480-5b27-11eb-9160-fb00096697ee.png)
- depth-wise conv와 1x1 projection layer 뒤에 1x1 expansion convolution으로 구성
- MnasNet[43]은 MobileNetV2 구조를 기반으로 하는 구조이며, bottleneck 구조에 squeeze and excitation에 기반한 모듈을 제안함.
  * squeeze and excitation module은 [20]의 ResNet 기반의 모듈에 다른 위치에서 integrated된 모듈
- MobileNetV3에선 이러한 방법들의 조합을 building block으로 사용하며, 이는 더 효율적인 모델을 만들기 위함

## Network Search
- MobileNetV3에선 각 network block을 최적화하여 global network structure를 찾았으며, platform-aware NAS를 사용
### Platform-Aware NAS for Block-wise Search
  -  global network structure를 찾기 위해 platform-aware neural architecture approach를 사용
### NetAdapt for Layer-wise Search
  - 아키텍처 탐색에서 상용된 두번째 기술인 NetAdapt

## Network Improvements
- 네트워크 초기와 끝부분의 연산비용이 비싼 레이어들을 재디자인함
- 새로운 nonlinearity인 h-swish를 제안
  * h-swish는 swish nonlinearity의 modified version으로 연산하기 더 빠르고 양자화-친화적임(quantization-friendly)
### Redesigning Expensive Layers
- 효율적인 last stage로 인해 latency를 7ms를 줄일 수 있었으며, 이는 전체 running time의 11%
![제목 없음1](https://user-images.githubusercontent.com/47103479/105129405-58b4de80-5b28-11eb-9515-9823bca3e979.png)
### Nonlinearities
### Large squeeze-and-excite
### MobileNetV3 Definitions
- MobileNetV3은 두 개의 모델로 정의됨
    * MobileNetV3-Large
    * MobileNetV3-Small
    
## Experiment
- classification, detection, segmentation으로 MobileNetV3의 성능 검증
  * Classification : Accuracy, latency, multiply adds (MAdds) 측정
  * Detection
    * MobileNetV3을 SSDLitye[39]의 backbone으로 사용
  * Semantic Segmentation
  
## Conclusion and Future Work
- MobileNetV3 Large와 Small 모델을 제안
- 차세대 모바일용 모델을 제안하기 위해 네트워크 설계 뿐만 아니라 여러 NAS 알고리즘들을 활용
