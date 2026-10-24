"""Inspect token batch shapes from your CSV/JSONL files and local tokenizer."""

import argparse
import json

from emoperso.datasets import iter_batches, load_dataset
from emoperso.deepseek_tokenizer import TokenizerConfig, load_tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',nargs='+',required=True,help='All cohort CSV/JSONL files to validate together')
    parser.add_argument('--tokenizer',required=True,help='Local DeepSeek-V3 tokenizer directory (no model weights)')
    parser.add_argument('--partition',choices=('train','validation','test','inference'),required=True)
    parser.add_argument('--layout',choices=('users','posts'),default='users')
    parser.add_argument('--order',choices=('source','recent'),default='source')
    parser.add_argument('--preprocess',action='store_true',help='Clean URLs, explicit MBTI labels and deleted-text markers')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--temp-dir',default=None,help='Writable private scratch directory')
    args=parser.parse_args()
    tokenizer=load_tokenizer(TokenizerConfig(tokenizer_name_or_path=args.tokenizer,revision=None,local_files_only=True))
    count=0
    with load_dataset(args.data,partition=args.partition,layout=args.layout,order=args.order,
                      preprocess=args.preprocess,temp_dir=args.temp_dir) as dataset:
        for batch in iter_batches(dataset,tokenizer=tokenizer,batch_size=args.batch_size,
                shuffle=args.shuffle,seed=args.seed,require_supervised=args.partition!='inference',temp_dir=args.temp_dir):
            count+=1
            print(json.dumps({'batch':count,'users':len(batch['user_keys']),
                              'input_shape':list(batch['input_ids'].shape),
                              'labels_shape':None if batch['labels'] is None else list(batch['labels'].shape)}))


if __name__=='__main__':main()
