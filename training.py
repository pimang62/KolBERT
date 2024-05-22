## https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-03/blob/main/colbert/train.py

## export CUDA_VISIBLE_DEVICES=2,3 실행 후

import pandas as pd
import argparse
from typing import List, Dict, Union
from datasets import DatasetDict, load_from_disk
from ragatouille import RAGTrainer
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="주어진 데이터로 ColBERT를 학습시킵니다.")
    parser.add_argument('--data_path', type=str, default='raw_data/baemin_qc.tsv', help='Documents 경로')
    parser.add_argument('--model_name_or_path', type=str, default="monologg/kobert", help='사전 학습 모델 또는 경로')
    return parser.parse_args()

def train(args: argparse.Namespace) -> None:
    """학습 데이터셋과 위키피디아 문서를 사용해 ColBERT를 학습시킵니다. 
    Args:
        args (argparse.Namespace): CLI에서 입력받은 학습 관련 파라미터 및 파일 경로 
    args.data_path:
        pd.DataFrame({"title":[str], "question":[str], "content" or "asnwer":[str]})
    args.model_name_or_path:
        pre_trained model name or path
        - default > "monologg/kobert"
        - after training > ".ragatouille/colbert/none/2024-04/18/10.49.10"
    """
    # Loading raw data
    dataset: pd.DataFrame = pd.read_csv(args.data_path, sep='\t')

    # Document Corpus
    corpus: List[str] = []
    for _, sample in tqdm(dataset.iterrows(), desc="Collecting corpus..."):
        title = sample["title"].strip()
        text = sample["content"].strip() if "content" in dataset.columns else sample["answer"].strip()
        if text not in corpus:  # 중복 방지
            string = title + ' [SEP][CLS] ' + text  # title + ' [SEP][CLS] ' + text
            corpus.append(string)

    # Making pairs (## triple)
    pairs: List[Union[str]] = [(sample["question"].strip(), sample["content"].strip() if "content" in dataset.columns else sample["answer"].strip()) \
                               for _, sample in tqdm(dataset.iterrows(), desc="Collecting queries, answers...")]
    ## triples: List[Union[str, int]] = [[example['question'], example['context'], 1] for example in dataset]  # 1 for pos, 0 for neg

    # Trainer ibject
    trainer = RAGTrainer(model_name="my_kolbert", pretrained_model_name=args.model_name_or_path, language_code="ko")

    # Hard Negative Passage using another embedding
    trainer.prepare_training_data(
        raw_data=pairs,  ## or triples included negative passage
        all_documents=corpus,
        mine_hard_negatives=True,
        hard_negative_model_size="large",
        pairs_with_labels=False  ## True
    )

    trainer.train(
        batch_size=128,
        nbits=4,
        maxsteps=100000000,  # not epoch just step
        use_ib_negatives=True,
        dim=128,  # 8 단위로
        learning_rate=5e-6,
        # doc_maxlen=100,  # 384
        use_relu=False,
        warmup_steps="auto",
    )


if __name__ == '__main__':
    args = parse_args()
    train(args)