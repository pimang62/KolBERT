# Retrieval Trainer (ColBERT)

Retrieval 모델 훈련을 위한 Repo 

### How To Run

* Data Format for Training
  * pd.DataFrame({"title":[str], "question":[str], "content" or "asnwer":[str]})

* Data Format for Indexing
  * txt, pdf, docx anyway (now pd.DataFrame only)

1. Create vertual environment
```
conda create -n kolbert python=3.9
pip install -r requirements.txt
```

2. Training
- data_path : 훈련용 데이터 경로
  - title, question, content(문단) or answer(문장) column이 존재하는 dataframe 형태여야 합니다. (현재)
  - dataset 객체나 corpus 객체를 만드는 부분을 수정하여 원하는 데이터 형태로 수정할 수 있습니다.
  - raw_data는 크게 pairs와 triples 형태로 제작하여 넣어줄 수 있습니다.
    - pairs : List or tuple > [postive, negative]
    - triples : List or tuple > [positive, negative, rank(ex. 1)]
- model_name_or_path : pre-trained model name or path
  - Default model name(or path)는 "monologg/kobert"이며, BERT/RoBERTa base의 huggingface_hub 경로를 입력해주어도 됩니다.

```
python training.py > logs/train.log
```

cf. wandb login 계정 token 입력 > colbert/training/training.py [line 88] : : wandb.init() project name과 entity 계정 이름을 변경하세요.


4. Indexing
- data_path : 문서 데이터 (현재는 training 데이터와 동일)
- model_name_or_path : 훈련시킨 모델의 경로를 넣어주시면 됩니다.
  > ex. .ragatouille/ 디렉토리 아래 존재합니다.
- index_root : 인덱스를 저장할 디렉토리명
- index_name : 인덱스 파일명
```
python indexing.py > logs/index.log
```

5. Searching
- data_path : 인덱싱 한 문서 데이터(여야 합니다.)
  - 평가를 내기 위한 목적
- index_path : 인덱싱 된 저장 경로
- topk : 검색할 문서 개수
```
python searching.py > logs/search.log
```

* Output
```
Query : 매출 정산은 언제 이루어지나요?
# 문서 1, 점수 28.546875
처음 배달의민족 광고를 신청하는 것이라면 [배민외식업광장 > 배민광고시작하기]를 통해 신청할 수 있습니다.

# 문서 2, 점수 28.46875
[이용방법] 1) 배민외식업광장 > 배민셀프서비스 > 주문 정산 > 부가세 신고 내역 2) 조회기간 설정 > 이메일 보내기 이메일 주소를 작성란에 받으실 분의 이메일 주소를 입력해 주세요.

# 문서 3, 점수 28.46875
[이용방법] 1) 배민외식업광장 > 셀프서비스 > 주문 정산 > 부가세 신고 내역 2) 조회기간 설정 > 이메일 보내기 이메일 주소를 작성란에 받으실 분의 이메일 주소를 입력해 주세요.

# 문서 4, 점수 28.40625
[배민스토어 고객센터] ☎1600-0025 [배민외식업광장> 배민비즈니스 > 배민스토어]에서 온라인으로 상담 신청을 하실 수도 있습니다.

# 문서 5, 점수 28.375
메뉴 모음컷 등록에 대한 자세한 내용은 배민외식업광장 이용가이드에서 확인해주세요. ([이용가이드] 메뉴모음컷 설정)

...

   question_index           retrieve_indices  rank  1/rank  isin
0               0  [149, 154, 321, 125, 324]     0     0.0     0
1               1    [279, 202, 16, 130, 69]     0     0.0     0
2               2    [205, 68, 117, 288, 21]     0     0.0     0
3               3  [113, 289, 177, 254, 206]     0     0.0     0
4               4   [52, 125, 324, 149, 260]     0     0.0     0
5               5  [289, 113, 177, 254, 128]     0     0.0     0
6               6   [321, 52, 279, 149, 260]     0     0.0     0
7               7   [205, 68, 254, 288, 100]     0     0.0     0
8               8  [117, 100, 220, 102, 205]     0     0.0     0
9               9   [113, 28, 125, 289, 324]     0     0.0     0
MRR score is 0.01178861788617886
Accuracy is 0.024390243902439025
```

### ToDo

- [ ] 

### Reference 
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [Ragatouille](https://github.com/bclavie/ragatouille)

### License 

