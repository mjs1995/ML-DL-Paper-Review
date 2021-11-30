# 협업필터링에서 포괄적 성능평가 모델(2012) - 유 석 종, 한국컴퓨터정보학회
- 내용기반 추천(content-based recommendation)
  - 개인화된 추천 방법에는 고객 프로필에 근거 
  - 사용자 프로필에 추천에 필요한 충분한 자료가 포함되어 있어야 하는 특성이 있고 상품과 사용자의 수가 증가할수록 확장성의 문제가 발생
  - 사용자 프로필과 아이템의 속성정보 간의 연관성(나이, 성별, 위치, 관심분야, 구매내역 등)을 바탕으로 이루어짐
  -  포트폴리오 효과(portfolio effect)를 유발하기 쉬운데, 즉 사용자가 이미 알고 있거나, 알고 있는 것과 유사한 아이템만을 주로 추천해 주
는 것

- 많이 구매되는 상품을 추천하는 통계적 방법(statistical recommendation)

- 협업필터링(collaborative filtering)
  - 사회형 추천 알고리즘으로 고객과 구매 성향이 유사한 집단을 찾아 선호할만한 미경험 아이템을 추천
  - 문제점
    - 협업필터링은 상품 평가 기록의 부족으로 인한 희박성(sparsity) 
      - 일반적으로 고객들은 상품 평가를 잘 하지 않는 경향이 있으며, 평가행렬이 희박할 경우 추천의 질을 떨어뜨림
    - 평가기록이 없는 새로운 고객과 상품에 대한 cold-start의 문제점
      - 평가를 전혀 하지 않은 새로운 사용자나 평가된 적이 없는 아이템은 협업필터링을 통해서 추천되기 어려움
    - Scalability
      - 협업필터링은 사용자와 아이템의 수가 증가할수록 유사 이웃 탐색에 필요한 연산 비용도 비례하여 증가
    - 추천 대상자의 변동에 따라 상품 예측오차에 편차가 발생하며 비일관된 추천 성능
    - 단일 알고리즘일수록 이 현상이 심화되는 경향
  - 평가 방법
    - 정확도 측정방법과 달리 추천 알고리즘이 포괄할 수 있는 아이템의 범위(coverage) 측정방법
    - 아이템 평가 자료의 수 대비 추천 질을 나타내는 학습율(learningrate) 평가방법
    - 인기 있는 제품 이외에 새롭고 참신한 상품을 추천할 수 있는 능력(novelty and serendipity) 평가방법
    - 추천 자신감(confidence) 평가 방법
    - 사용자 평가방법(user evaluation) 등
  
  ![image](https://user-images.githubusercontent.com/47103479/144067897-a7939cc4-be30-4d9e-9d7b-2a077ced6853.png)
  ![image](https://user-images.githubusercontent.com/47103479/144067921-11520f86-f777-43a2-ac0c-ceed2dd7abd3.png)
  
  ![image](https://user-images.githubusercontent.com/47103479/144067357-afc41cdd-1262-45f4-9a6b-3b92cd0519f0.png)
    
  - 내용기반 추천과는 대조적으로 협업필터링은 개별 아이템간의 유사도에 의존하는 것이 아니라 사용자 평가 유사도에 기반하고 있기 때문에 여러 장르에 걸친(cross-genre), 예상
하지 못한(serendipitous) 아이템들의 추천이 가능
  - 유사도 계산 방법에는 Pearson correlation coefficient, cosine similarity, mean square difference와 spearman correlation이 있으며, 이중 Pearson correlation coefficient가 가장 널리 사용

- 이웃탐색방법
  - k-Nearest Neighbor(kNN) 탐색은 목표 사용자와 전체 사용자간의 선호 유사도를 계산한 후 유사도 상위 k명을 이웃으로 선정하는 방법

- 성능평가 모델
  - 보다 정확한 CF알고리즘의 평가를 위하여 기존 지표인 MAE, MAE 편차, Precision, Recall을 포괄적으로 반영할 수 있는 확장형 성능평가 방법으로
MPRd(MAE, Precision, Recall) 모델을 제안

![image](https://user-images.githubusercontent.com/47103479/144068749-9ef211fe-4766-44ea-bb0f-049c39a714c5.png)
![image](https://user-images.githubusercontent.com/47103479/144068969-71dfe700-dff1-47d2-926f-b0e49b543a4d.png)
![image](https://user-images.githubusercontent.com/47103479/144068679-7edf93fb-837c-4b85-94f5-f336a65af380.png)
