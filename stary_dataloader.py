import json
import logging
from typing import Any,  Dict, List, Optional, Tuple
import os
import spacy
import jsonlines

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp_models.coref.util import make_coref_instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

#from data_deal import get_tokenized_doc_and_cluster, get_tokenized_doc, get_entity_list, get_gold_cluster_token, load_jsonl, get_split_data_list


logger = logging.getLogger(__name__)


def get_tokenized_doc_and_cluster(doc_str, entity, basic_tokenizer):
    """
        相校原始 get_tokenized_doc 添加将label转换为 cluster功能
        doc_str input text
        entity [(start, end, name)]
        subword_tokenizer     AutoTokenizer 分词器
        basic_tokenizer       Spacy 分词器
    """
    # doc 每句的分词结果  doc_entities 每句相对索引的[(start, end, name)]  doc_sentence 每句的sentence
    doc, doc_entities, doc_sentence = basic_tokenize_doc_entity(doc_str, entity, basic_tokenizer)

    word_idx = -1

    sentence_match_result_list = []
    for sentence, sentence_entity in zip(doc, doc_entities):
        # sentence spacy.token.doc.Doc  sentence_entity [(start_index, end_index, name)] start_index end_index 是相对位置
        sentence_match_result = []
        if sentence_entity != []:
            # TODO 插入 start_idx, end_idx, name
            # sentence_match_result list( spacy-word-index label-name)
            # 匹配的是 [(word_start_index, word_end_index, name)]
            sentence_match_result = get_pair_of_spacy_word_and_entity(sentence, sentence_entity)
            # print("sentence_match_result ", sentence_match_result)
        sentence_match_result_list.append(sentence_match_result)

    return doc, doc_entities, doc_sentence, sentence_match_result_list


def get_tokenized_doc(doc):
    # doc list[list]  subword_tokenizer AutoTokenizer
    tokens_cluster = []
    tokens = []
    word_idx = -1
    for sentence in doc:
        # 抽取一个句子
        orig_tokens = []
        for word in sentence:
            # 在每个句子中抽取单词
            orig_tokens.append(word)  # orig_tokens 新增加一个词
            tokens.append(word)
            # tokens 加一个word
        tokens_cluster.append(orig_tokens)

    return tokens_cluster, tokens


def get_global_index(paragraph_list, sentence_list, para_ind, sent_ind):
    """
        得到当前句的 start_index
    """
    temp_text = "\n\n".join(paragraph_list[:para_ind])
    temp_sent_text = '\n'.join(sentence_list[:sent_ind])

    if sent_ind == 0:
        #  段落中第一句
        if para_ind > 0:
            temp_text += '\n\n'
    else:
        #  非段落第一句
        if para_ind > 0:
            temp_text += '\n\n' + temp_sent_text + '\n'
        else:
            temp_text += temp_sent_text + '\n'
    return len(temp_text)


def basic_tokenize_doc_entity(doc_str, entities, basic_tokenizer):
    """
        # 前序 已经按照spacy 分句
        doc_str: 输入的Text
        entities [(start, end, name)]
        basic_tokenizer: Spacy 分词器
    """
    doc, doc_entities, doc_sentence, global_start, global_end, entity_index = [], [], [], 0, 0, 0
    paragraph_list = doc_str.split('\n\n')  # 将Text 按照段落做切分
    for para_idx, paragraph in enumerate(paragraph_list):
        sentence_list = paragraph.split('\n')  # 段落按照句子来做切分
        for sent_idx, sentence in enumerate(sentence_list):
            # para_idx 段落索引  sent_idx 句子索引   global_index 当前句的 start_index
            global_index = get_global_index(paragraph_list, sentence_list, para_idx, sent_idx)
            # print(" all length of ", doc_str[global_index], sentence, global_index)
            # sentence_entities  每句相对位置 [start, end, name]
            sentence_entities = get_inner_label(global_index, global_index + len(sentence), entities)
            doc.append(basic_tokenizer.tokenizer(sentence))  # 添加每句 分词id
            doc_entities.append(sentence_entities)  # 添加每句 相对位置的 Entity
            doc_sentence.append(sentence)  # 添加每句

            if sentence_entities != []:
                for start, end, name in sentence_entities:
                    entity_index += 1
                    # print(sentence[start:end] + " -------> " + name)
                    # print(sentence)

    # print("total length ", len(entities), entity_index)

    return doc, doc_entities, doc_sentence


def get_inner_label(global_start, global_end, entity):
    """
        得到 句子内部的 entity 相对坐标
    """
    sentence_entities = []

    for i in range(len(entity)):
        label_start, label_end, name = entity[i]["start_offset"], entity[i]["end_offset"], \
                                       entity[i]["label"]

        if entity[i]["start_offset"] >= global_start and entity[i]["end_offset"] <= global_end:
            sentence_entities.append([label_start - global_start, label_end - global_start, name])

    return sentence_entities

def get_pair_of_spacy_word_and_entity(sentence, entity):
    word_list = [str(word) for word in sentence]  # 将 spacy 分词的结果提出
    sentence_text = str(sentence)  # sentence_text
    sentence_match_result = []
    # word_list_start_index [(start, end)] 每一个词的 在 text 中的 start end 索引
    word_list_start_index = get_begin_pos_of_spacy_word_from_sentence(word_list, sentence_text)
    for start_index, end_index, name in entity:
        # 该 label 对应的 [[word_ind, name]] word_ind 为 sentence内部的索引
        word_match_result = get_entity_spacy_word(start_index, end_index, word_list_start_index, name, sentence_text)
        sentence_match_result.append(word_match_result)
    # 匹配的是 [(word_start_index, word_end_index, name)]
    sentence_match_result = merge_sentence_match_result(sentence_match_result)
    return sentence_match_result


def get_begin_pos_of_spacy_word_from_sentence(word_list, sentence_text):
    begin_index, word_index = 0, 0
    word_list_start_index = []
    while (
            (begin_index < len(sentence_text)) and
            word_index < len(word_list)
    ):
        word = word_list[word_index]
        sentence_first = sentence_text[:begin_index]
        sentence_second = sentence_text[begin_index:]
        start_index, end_index = count_subword_begin(word, sentence_second)
        if start_index is None:
            raise ValueError
        else:
            word_list_start_index.append([len(sentence_first) + start_index, len(sentence_first) + end_index])
            begin_index += end_index
            word_index += 1

    assert len(word_list) == len(word_list_start_index)
    return word_list_start_index


def count_subword_begin(subword, text):
    if len(subword) > len(text):
        return None, None
    for i in range(len(text) - len(subword) + 1):
        if subword == text[i:i + len(subword)]:
            return i, i + len(subword)
    return None, None


def get_entity_spacy_word(start_index, end_index, word_list_start_index, name, sentence_text):
    # start_index, end_index label 在 sentence 的相对位置
    word_match_result = []
    for ind, word_indexs in enumerate(word_list_start_index):
        word_start_index, word_end_index = word_indexs
        if verify_intersection(start_index, end_index, word_start_index, word_end_index):
            word_match_result.append([ind, name])

    assert len(word_match_result) >= 1
    return word_match_result


def verify_intersection(start_index, end_index, word_start_index, word_end_index):
    entity_part = [i for i in range(start_index, end_index)]
    word_part = [i for i in range(word_start_index, word_end_index)]
    for ent in entity_part:
        if ent in word_part:
            return True
    return False


def merge_sentence_match_result(sentence_match_result):
    merge_result = []
    for word_match_result in sentence_match_result:
        start_index, name, end_index = word_match_result[0][0], word_match_result[0][-1], word_match_result[-1][0]
        merge_result.append([start_index, end_index, name])
    return merge_result


def get_entity_list(tokens_cluster, sentence_match_result_list):
    sen_len = 0
    entity_list = []
    for senidx, sen in enumerate(tokens_cluster):
        # if len sentence_match_result_list[senidx]
        for entidx, ent in enumerate(sentence_match_result_list[senidx]):
            ent_dic = {}
            ent_dic['start_offset'] = ent[0] + sen_len
            ent_dic['end_offset'] = ent[1] + sen_len
            ent_dic['label'] = ent[2]
            ent_dic['span_pos'] = [ent[0] + sen_len, ent[1] + sen_len]
            entity_list.append(ent_dic)
        sen_len += len(sen)
    return entity_list


def get_gold_cluster_token(entity_tokenized):
    gold_cluster_dic = {}
    for idx, item in enumerate(entity_tokenized):
        if item['label'] in gold_cluster_dic:
            gold_cluster_dic[item['label']].append(item['span_pos'])
        else:
            gold_cluster_dic[item['label']] = [item['span_pos']]

    gold_cluster = []
    gold_cluster_label = []
    for key in gold_cluster_dic:
        gold_cluster.append(gold_cluster_dic[key])
        gold_cluster_label.append(key)

    gold_cluster = [[tuple([m[0], m[1]]) for m in gc] for gc in gold_cluster]
    gold_cluster = [clusters_ for clusters_ in gold_cluster if len(clusters_) > 1]

    return gold_cluster, gold_cluster_label, gold_cluster_dic


## data_split

allow_role_name = role_name = ['Savannah','Zack', 'Kyle', 'Zara', 'Kai', 'driver', 'guards', 'Elene',  'Gideon', 'The maid',  'Starlight',  'Other People', "Starlight's Parents", 'Lord Hales', 'Lady Hales',  'The Guard', 'The King',  'Valley', 'Alexandre', 'The orphans',  'The soon-to-be crown princess', 'Clara', 'Michelle', 'Mason', 'My Wolf', 'Deltas', 'Omegas', 'James', 'Melody', 'Artemis', 'Leo', 'Zero', 'Hero', 'Jess',
             'Mama', 'The Prince', 'Jayde', 'Jessabelle', 'Luca', 'Roxie', 'Anna Gabriel', 'The man', 'Harry', 'Mr.Wright', 'Gabe Lesley', 'The little girl', 'Isaac', 'men with batons', 'Lavinia Devon', 'stricken girl', 'Lavinia', 'forman', 'Rosie', 'Greenery', 'Astor kid', 'the six girls', 'Helen', 'Freya Winter', 'a male', 'Beaufort', 'the curly guy', 'the boy',  'Eliza', 'Cinderella', 'Beaufort family', 'girl with dark hair', 'a young man', 'professor']



def load_jsonl(jsonl_path):
    jsonl_content = []
    with open(jsonl_path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            if type(item) == type(jsonl_content):
                return item
            jsonl_content.append(item)
            entities = [ent for ent in item['entities'] if ent['label'] in allow_role_name]
            item['entities']  = entities
    return jsonl_content

# 将text分成n等份
def split_data_text(s, n):  # text: str
    fn = len(s) // n
    rn = len(s) % n
    ar = [fn + 1] * rn + [fn] * (n - rn)
    si = [i * (fn + 1) if i < rn else (rn * (fn + 1) + (i - rn) * fn) for i in range(n)]
    sr = [s[si[i]:si[i] + ar[i]] for i in range(n)]
    return sr


def split_data(data, n=2):
    text = data['text']
    entity = data['entities']
    text_split = split_data_text(text, n)
    split_len = len(text_split[0])

    entity_split = []
    entity_tmp1 = []
    entity_tmp2 = []

    for ent in entity:
        offset = ent['start_offset']
        start_offset_ = ent['start_offset'] % split_len
        end_offset_ = ent['end_offset'] % split_len

        ent_tmp = ent.copy()
        if start_offset_ <= end_offset_:
            ent_tmp['start_offset'] = start_offset_
            ent_tmp['end_offset'] = end_offset_

            if offset < split_len:
                entity_tmp1.append(ent_tmp)
            else:
                entity_tmp2.append(ent_tmp)

    entity_split.append(entity_tmp1)
    if (entity_tmp2):
        entity_split.append(entity_tmp2)

    return text_split, entity_split


def get_split_data(data):
    text_split, entity_split = split_data(data, n=2)

    data1 = {}
    data2 = {}
    data1['id'] = data['id']
    data1['text'] = text_split[0]
    data1['entities'] = entity_split[0]
    if (len(entity_split) == 1):
        return data1, data2

    data2['id'] = data['id']
    data2['text'] = text_split[1]
    data2['entities'] = entity_split[1]

    return data1, data2


def get_split_data_list(data_list):
    data_list_re = []
    for data in data_list:
        data1, data2 = get_split_data(data)
        data_list_re.append(data1)
        if data2:
            data_list_re.append(data2)
    return data_list_re


def extend_all_part(part_content):
    all_content = []
    for part in part_content:
        all_content.extend(part)
    return all_content


@DatasetReader.register("staryEval")
class StaryEvalReader(DatasetReader):


    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters


#####
    def _read(self, file_path: str):

        jsonl_content = [line for line in load_jsonl(file_path) if line["entities"] != []]

        #jsonl_content = get_split_data_list(jsonl_content)
        #jsonl_content = get_split_data_list(jsonl_content)
        #jsonl_content = get_split_data_list(jsonl_content)
        #jsonl_content = get_split_data_list(jsonl_content)

        basic_tokenizer = spacy.load("en_core_web_sm")
        for data in jsonl_content:

            doc, doc_entities, doc_sentence, sentence_match_result_list = get_tokenized_doc_and_cluster(data['text'], data['entities'],
                                                                                                        basic_tokenizer)
            tokens_cluster, text_tokenized = get_tokenized_doc(doc)
            entity_list = get_entity_list(tokens_cluster, sentence_match_result_list)
            gold_cluster, gold_cluster_label, gold_cluster_dic = get_gold_cluster_token(entity_list)
            text_tokenized = [str(x) for x in text_tokenized]
            yield self.text_to_instance([Token(x) for x in text_tokenized], gold_cluster)


    def text_to_instance(
            self,  # type: ignore
            sentence: List[Token],
            gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,

    ) -> Instance:
        
        metadata: Dict[str, Any] = {"original_text": sentence}
        
        #sentence = extend_all_part(sentence)
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField(sentence, self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
            "metadata": metadata_field,
        }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)


@DatasetReader.register("stary")
class StaryReader(DatasetReader):
    """
    Reads a single JSON-lines file for [the PreCo dataset](https://www.aclweb.org/anthology/D18-1016.pdf).
    Each line contains a "sentences" key for a list of sentences and a "mention_clusters" key
    for the clusters.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = `None`)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: `int`, optional (default = `None`)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = `False`)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them.
    """




    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters


#####
    def _read(self, file_path: str):

        jsonl_content = [line for line in load_jsonl(file_path) if line["entities"] != []]

        jsonl_content = get_split_data_list(jsonl_content)
        jsonl_content = get_split_data_list(jsonl_content)
        jsonl_content = get_split_data_list(jsonl_content)
        jsonl_content = get_split_data_list(jsonl_content)

        basic_tokenizer = spacy.load("en_core_web_sm")
        for data in jsonl_content:

            doc, doc_entities, doc_sentence, sentence_match_result_list = get_tokenized_doc_and_cluster(data['text'], data['entities'],
                                                                                                        basic_tokenizer)
            tokens_cluster, text_tokenized = get_tokenized_doc(doc)
            entity_list = get_entity_list(tokens_cluster, sentence_match_result_list)
            gold_cluster, gold_cluster_label, gold_cluster_dic = get_gold_cluster_token(entity_list)
            text_tokenized = [str(x) for x in text_tokenized]
            yield self.text_to_instance([Token(x) for x in text_tokenized], gold_cluster)


    def text_to_instance(
            self,  # type: ignore
            sentence: List[Token],
            gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,

    ) -> Instance:
        metadata: Dict[str, Any] = {"original_text": sentence}
        #sentence = [Token(x) for x in sentence]
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        
        text_field = TextField(sentence, self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": span_field,
            "metadata": metadata_field,
        }
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)




    

#dataset_reader = StaryReader(max_span_width = 20)
#instances = dataset_reader.read("../../../training_config/coref/all_chapter_2.jsonl")
#instances = dataset_reader.read("../../../training_config/coref/2375715.jsonl")
#for idx, item in enumerate(instances):
    #if idx ==9:
        #print(item.fields)
        #print(len(item['text']))

        
