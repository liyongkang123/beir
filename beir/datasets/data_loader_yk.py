from __future__ import annotations

import csv
import json
import logging
import os
from typing import Union, Iterable      # 添加

# from examples.retrieval.evaluation.custom.evaluate_custom_dataset_files import qrels_path
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class GenericDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {} # 原来是 {}, 现在是装着 dict 的 list

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    # def load_custom(
    #     self,
    # ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    #     self.check(fIn=self.corpus_file, ext="jsonl")
    #     self.check(fIn=self.query_file, ext="jsonl")
    #     self.check(fIn=self.qrels_file, ext="tsv")
    #
    #     if not len(self.corpus):
    #         logger.info("Loading Corpus...")
    #         self._load_corpus()
    #         logger.info("Loaded %d Documents.", len(self.corpus))
    #         logger.info("Doc Example: %s", list(self.corpus.values())[0])
    #
    #     if not len(self.queries):
    #         logger.info("Loading Queries...")
    #         self._load_queries()
    #
    #     if os.path.exists(self.qrels_file):
    #         self._load_qrels()
    #         self.queries = {qid: self.queries[qid] for qid in self.qrels}
    #         logger.info("Loaded %d Queries.", len(self.queries))
    #         logger.info("Query Example: %s", list(self.queries.values())[0])
    #
    #     return self.corpus, self.queries, self.qrels

    def load(self, split: str | list[str] = "test" ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

        # 2. 统一为列表，便于后续遍历
        if isinstance(split, str):
            splits: Iterable[str] = [split]
        else:
            splits = split

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")


        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self.queries_split = {}
        for sp_name in splits:
            qrels_file= os.path.join(self.qrels_folder, f"{sp_name}.tsv")
            if not  (os.path.exists(qrels_file)):
                raise ValueError(f"File {qrels_file} not present! Please provide accurate file.")
            # loader qrels directly because qrels_file is areadly exists
            qrels = self._load_qrels(qrels_file)
            self.queries_split.update( {qid: self.queries[qid] for qid in  qrels})
            self.qrels[sp_name]= qrels

        logger.info("Loaded %d %s Queries.", len(self.queries_split), split )
        logger.info("Query Example: %s", list(self.queries_split.values())[0])

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split )
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if len(self.qrels)==1:
            return self.corpus, self.queries_split, self.qrels[splits[0]]
        else:
            return self.corpus, self.queries_split, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self, qrels_file: str = ""):
        reader = csv.reader(
            open( qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)

        qrels= {}
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in  qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score
        return qrels
