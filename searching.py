## https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-03/blob/main/colbert/searching.py

import argparse
from datasets import load_from_disk, DatasetDict
from ragatouille import RAGPretrainedModel
from typing import List, Dict, Tuple, Any
import torch
from torch import tensor
import pandas as pd
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사전 학습된 모델을 사용해 쿼리와 관련있는 문서를 검색합니다.")
    parser.add_argument('--data_path', type=str, default='raw_data/baemin_qc.tsv',
                        help='데이터셋 저장 경로')
    parser.add_argument('--index_path', type=str, default='index/colbert/indexes/baemin',  # check in your 'index/...'
                        help='인덱스 작업을 마친 모델이 저장된 경로')
    parser.add_argument('--topk', type=int, default=5,
                        help='검색할 문서의 개수')
    return parser.parse_args()


def load_retrieval(args: argparse.Namespace) -> None:
    """사전 학습된 모델을 사용해 쿼리와 관련있는 문서를 검색합니다.

    Args:
        args (argparse.Namespace): CLI에서 입력받은 검색 작업에 필요한 파일 경로와 저장 경로
    """
    # Loading pre-trained model
    RAG = RAGPretrainedModel.from_index(
        index_path=args.index_path        
    )

    return RAG


def get_relevant_docs(retrieval, query: str) -> List[Dict]:
    """ColBERT 모델을 사용해 쿼리에 관련된 문서를 검색합니다. 
    Args:
        query : str

    Returns:
        List[Dict[str]]: 쿼리에 관련된 문서에 관한 딕셔너리 top-k 리스트
    """
    results: List[Dict] = retrieval.search(query=query, index_name="baemin", k=args.topk)
    print(f"Query : {query}")
    for i, res in enumerate(results, 1):
        print(f"# 문서 {i}, 점수 {res['score']}\n{res['content']}", end='\n')
        print()
    return results  # top-k


def calculate_rank(y_pred, y_true):
    """Calculate rank and inverse rank
       Return rank, 1/rank 
    """
    rank_index = torch.nonzero(y_pred==y_true, as_tuple=False)  # [[1]] index

    if not len(rank_index):   # 같은 인덱스가 하나도 없다면
        return 0, 0
    else:
        rank = rank_index.squeeze().item() + 1
        return rank, 1/rank


if __name__ == '__main__':
    args = parse_args()
    retrieval = load_retrieval(args)
    
    # Loading raw dataset
    dataset = pd.read_csv(args.data_path, sep='\t')
    
    retrieve_list = []
    rank_list = []
    inverse_rank_list = []
    isin = []

    for idx, (_, sample) in enumerate(dataset.iterrows()):
        query = sample["question"].strip()

        results: List[Dict] = get_relevant_docs(retrieval, query)

        retrieve_list.append([docs["passage_id"] for docs in results])

        y_pred = tensor([docs["passage_id"] for docs in results])  # tensor([77, 35, 127])
        y_true = tensor(idx)
    
        # rank와 1/rank 계산하기
        rank, inverse_rank = calculate_rank(y_pred, y_true)

        rank_list.append(rank) 
        inverse_rank_list.append(inverse_rank)
        isin.append(1 if idx in y_pred.tolist() else 0)

    df = pd.DataFrame({'question_index': range(len(retrieve_list)),
                    'retrieve_indices': retrieve_list,
                    'rank': rank_list,
                    '1/rank': inverse_rank_list,
                    'isin': isin})
    
    print(df.head(10))

    print(f"MRR score is {df['1/rank'].mean()}")
    print(f"Accuracy is {df['isin'].mean()}")