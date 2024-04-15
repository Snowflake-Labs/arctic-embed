import os
import json
import tqdm
import numpy as np
import torch
import argparse

from datasets import Dataset
import pdb
from typing import List, Dict
from functools import partial
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader
from mteb import MTEB

from eval.MTEB.utils import pool, logger, move_to_cuda

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='bert-base-uncased',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages')
parser.add_argument('--pool-type', default='cls', help='pool type')
parser.add_argument('--batch-size', default=1024, help='batch size')


args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg'], 'pool_type should be cls or avg'
assert args.output_dir, 'output_dir should be set'
os.makedirs(args.output_dir, exist_ok=True)


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    return tokenizer(examples['contents'],
                     max_length=512,
                     padding=True,
                     return_token_type_ids=False,
                     truncation=True)


class RetrievalModel():
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, , add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()
        self.batch_size = args.batch_size
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['Represent this sentence for searching relevant passages: {}'.format(q) for q in queries] #vs query: prompt
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['{}'.format(t) for t in input_texts] #No doc prefix
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({'contents': input_texts})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


def main():
    model = RetrievalModel()
    task_names = ['ArguAna', 'ClimateFEVER', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval', 'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackWordpressRetrieval', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID']
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:
        logger.info('Processing task: {}'.format(task))

        args.doc_as_query = task in ['QuoraRetrieval']

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test" if task not in ['MSMARCO'] else 'dev'],
                       output_folder=args.output_dir,
                       overwrite_results=False)


if __name__ == '__main__':
    main()
