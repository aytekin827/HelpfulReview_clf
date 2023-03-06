# HelpfulReview_clf
https://magical-erigeron-538.notion.site/Helpful-Review-Model-a29a7c792e434bfd82f4d182f814389a

## 프로젝트 개요
  1. 배경
  온라인 쇼핑몰에서 리뷰의 중요성은 엄청납니다.  
  소비자들은 판매자가 제공하는 정보 외에 다른 구매자들이 작성한 리뷰를 보고 추가 정보를 얻습니다. 그리고 이는 소비자들의 구매여부에 엄청난 영향을 끼치고 있습니다.  
  또 판매자는 소비자들의 리뷰를 통해서 자신이 판매하는 상품의 약점과 강점을 파악할 수 있게 되어 향후 사업 및 상품 개발을 위한 insight를 얻을 수 있습니다.  
  하지만 바쁜 현대인들에게 수많은 리뷰들을 모두 읽어볼 시간은 없습니다. 리뷰들 중에서 어떤 리뷰가 내가 구매하고자 하는 상품과 또 이 상품이 속한 카테고리에서 도움이 되는 리뷰인지 알 수 있다면 온라인 쇼핑 시장에서 소비자와 판매자에게 큰 도움이 될 것이라고 생각합니다.   
  즉, 쌓여가는 리뷰들 사이에서 핵심 리뷰를 쪽집게처럼 찝어주는 모델을 만들기 위해서 이 프로젝트를 시작하게 되었습니다.

  2. 목적  
  상품의 카테고리별 도움이 되는 리뷰 분류기 제작
  
  3. 기대효과  
    - 소비자 :  
        도움이 되는 리뷰를 소비자에게 제공하여 합리적인 소비가 가능함.  
        리뷰 작성 중 즉각적인 피드백을 통해서 도움이 되는 리뷰를 적도록 유도할 수 있음.  
    - 판매자 :    
        도움이 되는 리뷰를 통해서 사업과 상품개발의 insight를 얻을 수 있음.
      
      
## 데이터
- 데이터 수집
  자체 제작 커스텀 크롤러를 이용하여 바지/레깅스(47578개), 스포츠의류(25038개), 후드집업(20195개) 이렇게 세가지 카테고리의 리뷰데이터를 수집하였습니다.
  
 - EDA
  리뷰제목(review_headline), 리뷰(review), 상품명(prod_name), 카테고리명(category_name), 도움이 됐어요 개수(help_count) 총 5가지 특성을 크롤링 하였습니다.  
  
  <img src="https://user-images.githubusercontent.com/78078141/215737337-f5d3dab4-678a-4162-89a0-fd96ff9a8e57.png" width="50%" height="50%"/>
 
 - 워드 클라우드
  **Okt형태소분석기**와 counter라이브러리를 이용하여 워드 클라우드를 만들어 보았다.

  당연히 바지/레깅스 카테고리의 리뷰이기 때문에 바지의 글자가 가장 크게 나타났다. 

  소비자들이 주로 사이즈와 길이, 허리, 가격을 중요하게 생각해서 리뷰에 많이 작성했다는 것을 알 수 있다.
  
  <img src="https://user-images.githubusercontent.com/78078141/215738615-c908c3a4-a2c0-4cf9-a8f9-aa347a62a19b.png" width="50%" height="50%"/>
  
  - 라벨링
  **‘도움이 됐어요'** 가 **1개 이상**인 데이터(label 1) → **도움이 되는** 리뷰(**helpful** review)  
  **’도움이 됐어요'** 가 **0개**인 데이터(label 0) → **도움이 되지 않는** 리뷰(**helpless** review)  

  라벨의 비율이 대략 50:50이다.

  <img src="https://user-images.githubusercontent.com/78078141/215740015-0cc3889a-1bbd-4942-8aa5-90252298146b.png" width="50%" height="50%"/>
  
  
 ## 모델링
 1. KcELECTRA를 사용한 분류모델
    
    Pre-Trained model : **KcELECTRA (**https://github.com/Beomi/KcELECTRA**)**
    
    KcELECTRA모델은 **온라인뉴스에서 댓글과 대댓글 데이터**를 학습한 모델로 뉴스, 위키, 책, 법조문과 같이 잘 정제된 텍스트가 아닌 **구어체와 신조어, 오탈자 등 정제되지 않은 텍스트 분석**에 적합합니다. 
    
    **Huggingface의 Transformers 라이브러리**를 통해서 간편히 불러와서 사용가능합니다.
    
    Fine tuning : 각 카테고리별 쿠팡 리뷰 데이터(바지레깅스-47578, 스포츠의류-25038 ,후드집업-20195)
    
    training result
    
    바지레깅스 : Train acc = 0.7418, Val acc =0.6955 , Test acc = 0.6975

    스포츠의류 : Train acc = 0.8967 , Val acc = 0.7038, Test acc = 0.7068

    후드집업 : Train acc = 0.8671, Val acc = 0.6720 , Test acc = 0.6907
    
  ## tkinter를 이용한 GUI 제작
  - **제품 카테고리와 상품명, 첨부파일개수, 리뷰**를 **입력**하여 추론(inference)합니다.
- **확률**과 함께 **추론 결과를 출력**합니다.

  <img src="https://user-images.githubusercontent.com/78078141/215740355-a76da9b6-d2eb-42ae-98c8-42b8edf047e0.png" width="50%" height="50%"/>


### reference
[https://github.com/Beomi/KcELECTRA](https://github.com/Beomi/KcELECTRA)

[PyTorch documentation - PyTorch 1.13 documentation](https://pytorch.org/docs/stable/index.html)

[beomi/KcELECTRA-base · Hugging Face](https://huggingface.co/beomi/KcELECTRA-base)

 
