## https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-03/blob/main/colbert/indexing.py

import pandas as pd
import argparse
from ragatouille import RAGPretrainedModel
from typing import Any, List
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColBERT를 사용해 위키피디아 문서에 인덱싱 작업을 수행합니다.")
    parser.add_argument('--data_path', type=str, default='raw_data/baemin_qc.tsv',
                        help='baemin documents json 경로')
    parser.add_argument('--model_name_or_path', type=str, default=".ragatouille/colbert/none/2024-05/02/11.51.36/checkpoints/colbert",  # check in your '.ragatouille/...',
                        help='사전 학습 모델 또는 경로')
    parser.add_argument('--index_root', type=str, default='index',
                        help='구축된 인덱스를 저장할 경로')
    parser.add_argument('--index_name', type=str, default='baemin',
                        help='생성할 인덱스 파일의 이름')
    return parser.parse_args()

def index(args: argparse.Namespace) -> None:
    """사전 학습된 ColBERT embedding을 사용해 Index를 부여합니다.

    Args:
        args (argparse.Namespace): CLI에서 입력받은 파일 경로와 모델 정보
    """
    # Loading raw data
    dataset = pd.read_csv(args.data_path, sep='\t')
    
    # Documents Corpus
    corpus: List[str] = []
    for _, sample in tqdm(dataset.iterrows(), desc="Collecting corpus..."):
        title = sample["title"].strip()
        text = sample["content"].strip() if "content" in dataset.columns else sample["answer"].strip()
        if text not in corpus:  # 중복 방지
            string = title + ' [SEP][CLS] ' + text  # title + ' [SEP][CLS] ' + text
            corpus.append(string)

    # Loading pre-trained ColBERT model
    RAG = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        index_root=args.index_root
    )
    
    # Indexing
    RAG.index(
        collection=corpus,
        index_name=args.index_name,
        max_document_length=100,
        split_documents=True
    )

if __name__ == '__main__':
    args = parse_args()
    index(args)