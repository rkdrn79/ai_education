from datasets import load_dataset
from src.dataset.dataset import LLMDataset
from src.dataset.datacollator import LLMDataCollator

from cv2 import imread


def get_dataset(args, tokenizer):
    # Load the dataset
    instruction = """#### LLM을 위한 지침: 
당신은 악성 도박 광고를 탐지하고 처리하도록 설계된 시스템입니다. 당신의 임무는 광고의 텍스트를 정제하고 이미지 분석에 기반하여 중심 위치를 탐지하는 것입니다. 다음 단계에 따라 출력을 생성하세요.

#### 입력: 
1. **텍스트**: 대전동부경찰서논 강노살인 형의로 A(5O대)씨루 긴급체포해 조사하고 있다고 10일 밝화다. 북극해에 나타난 오징어: ."현실이 된 기후위 기" A씨는 지난 5일 대천 동구 성남동에서 불법도박장으로 운영되던 한 단독주택에서 도박물 하다 도박장 업주인 B(6O대) 씨틀 목 콜라 살해한 뒤 달아난 형의틀 반고 있다 '형기채우면 100살 넘어' . 배우자 실해 70.80 대 중형 잇따라 스코어 맞추면 안되다 / 국내 스코어 역배당 적중률 99.9% 체험금으로 적중시 현금바로인출 사이트는 곧 폭발적인 회원가입 예상되니다. / 카카오록 cocamouse AUVERTISEMEAT ADVERTSEMEhT 신고들 받은 경찰은 추적 끝에 전날 부산 해운대 한 숙박업소메서 A씨률 긴급체포하다 최초
2. **이미지**: <image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>  # 이미지의 비전 토큰이 16개로 반복됨
3. **메타데이터**: {0: {'center_postion': array([799.5, 537.5]), 'text': 'II에 [태)내 JIIIUIDISIM2I'}, 1: {'center_postion': array([799.5, 468.5]), 'text': '슬못 그동에그계면 스편 카지노 런칭 신규 2096 첫중 1'}, 2: {'center_postion': array([799.5, 394. ]), 'text': '{표팔과 3'}, 3: {'center_postion': array([456.25, 406.75]), 'text': 'Adobe Stock 30일간 무료로 체험해 보세요. 무료 기법하가 Auooe'}, 4: {'center_postion': array([190. , 406.5]), 'text': 'POKER ONLINE'}, 5: {'center_postion': array([799.75, 320.25]), 'text': '국내 1동 먹뒤감중 전문 커유니터 인중업체 깨리스로 최대 3천만원'}, 6: {'center_postion': array([799.75, 249.  ]), 'text': '%@'}, 7: {'center_postion': array([325. , 177.5]), 'text': 'SAIUUWT UIVEIA: 오다카사노 젓출20% 태기미h nA 3내지스무? 매춘104 30+ 50*'}}

#### 작업: 
1. **텍스트 정제**: 주어진 텍스트를 처리하여 도박 관련 내용을 제거합니다. 
2. **중심 위치 탐지**: 이미지의 비전 토큰과 제공된 메타데이터를 분석하여 악성 도박 광고의 중심 위치를 결정합니다. 
3. **출력 형식**: 출력에는 정제된 텍스트와 탐지된 악성 광고의 중심 위치가 포함되어야 합니다. 

#### 출력 형식: 
```json
{
  "clean_text": "<도박 관련 내용이 제거된 텍스트>",
  "ad_center_position": "<악성 도박 광고의 중심 위치>"
}
"""

    image = imread("/home/work/competition/ai_education/KakaoTalk_Photo_2024-10-12-04-03-40.png")
    
    train_ds = LLMDataset(args, instruction, image)
    valid_ds = LLMDataset(args, instruction, image)
    datacollator = LLMDataCollator(args)

    return train_ds, valid_ds, datacollator