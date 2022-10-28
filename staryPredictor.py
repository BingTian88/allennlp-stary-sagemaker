from typing import List, Dict


from spacy.tokens import Doc
import numpy

from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SequenceLabelField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token
import spacy

def basic_tokenize_doc(doc_str, basic_tokenizer):
    """
        # 前序 已经按照spacy 分句
        doc_str: 输入的Text
        entities [(start, end, name)]
        basic_tokenizer: Spacy 分词器
    """
    doc =  []
    paragraph_list = doc_str.split('\n\n')  # 将Text 按照段落做切分
    for para_idx, paragraph in enumerate(paragraph_list):
        sentence_list = paragraph.split('\n')  # 段落按照句子来做切分
        for sent_idx, sentence in enumerate(sentence_list):
            doc.append(basic_tokenizer.tokenizer(sentence))  # 添加每句 分词id          
    return doc

def get_tokenized_doc(doc):
    # doc list[list]  subword_tokenizer AutoTokenizer
    tokens = []
    for sentence in doc:   
        for word in sentence:
            tokens.append(word)
    return tokens

def extend_all_part(part_content):
    all_content = []
    for part in part_content:
        all_content.extend(part)
    return all_content


@Predictor.register("core")
class CorefPredictorStary(Predictor):
    """
    Predictor for the [`CoreferenceResolver`](../models/coreference_resolution/coref.md) model.

    Registered as a `Predictor` with name "coreference_resolution".
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenize our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)

    def predict(self, document: str):
        
        doc = basic_tokenize_doc(document, self._spacy)
        doc = get_tokenized_doc(doc)
        doc = [str(x) for x in doc]

        instance = self._dataset_reader.text_to_instance([Token(x) for x in doc])
        return self.predict_instance(instance)
     

    def predict_tokenized(self, tokenized_document: List[str]) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        # Parameters

        tokenized_document : `List[str]`
            A list of words representation of a tokenized document.

        # Returns

        A dictionary representation of the predicted coreference clusters.
        """
        instance = self._words_list_to_instance(tokenized_document)
        return self.predict_instance(instance)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        Takes each predicted cluster and makes it into a labeled `Instance` with only that
        cluster labeled, so we can compute gradients of the loss `on the model's prediction of that
        cluster`.  This lets us run interpretation methods using those gradients.  See superclass
        docstring for more info.
        """
        # Digging into an Instance makes mypy go crazy, because we have all kinds of things where
        # the type has been lost.  So there are lots of `type: ignore`s here...
        predicted_clusters = outputs["clusters"]
        span_field: ListField = instance["spans"]  # type: ignore
        instances = []
        for cluster in predicted_clusters:
            new_instance = instance.duplicate()
            span_labels = [
                0 if (span.span_start, span.span_end) in cluster else -1  # type: ignore
                for span in span_field
            ]  # type: ignore
            new_instance.add_field(
                "span_labels", SequenceLabelField(span_labels, span_field), self._model.vocab
            )
            new_instance["metadata"].metadata["clusters"] = [cluster]  # type: ignore
            instances.append(new_instance)
        if not instances:
            # No predicted clusters; we just give an empty coref prediction.
            new_instance = instance.duplicate()
            span_labels = [-1] * len(span_field)  # type: ignore
            new_instance.add_field(
                "span_labels", SequenceLabelField(span_labels, span_field), self._model.vocab
            )
            new_instance["metadata"].metadata["clusters"] = []  # type: ignore
            instances.append(new_instance)
        return instances
    
    
    ###improved resolution text
    # ignore (we don't resove them at all) the clusters that doesn't contain any noun phrase
    def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
        spans = [doc[span[0]:span[1]+1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
        return span_noun_indices

    # pick the first noun phrase in the cluster as cluster head
    def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end+1]
        return head_span, [head_start, head_end]

    def improved_replace_corefs(document, clusters):
        resolved = list(tok.text_with_ws for tok in document)

        for cluster in clusters:
            noun_indices = get_span_noun_indices(document, cluster)

            if noun_indices:
                mention_span, mention = get_cluster_head(document, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention:  # we don't replace the head itself
                        core_logic_part(document, coref, resolved, mention_span)

        return "".join(resolved)


    def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        final_token = document[coref[1]]
        if final_token.tag_ in ["PRP$", "POS"]:
            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[coref[0]] = mention_span.text + final_token.whitespace_
        for i in range(coref[0] + 1, coref[1] + 1):
            resolved[i] = ""
        return resolved
    
     ###improved resolution text
    

    @staticmethod
    def replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
        """
        Uses a list of coreference clusters to convert a spacy document into a
        string, where each coreference is replaced by its main mention.
        """
        # Original tokens with correct whitespace
        resolved = list(tok.text_with_ws for tok in document)

        for cluster in clusters:
            # The main mention is the first item in the cluster
            mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
            mention_span = document[mention_start:mention_end]

            # The coreferences are all items following the first in the cluster
            for coref in cluster[1:]:
                final_token = document[coref[1]]
                # In both of the following cases, the first token in the coreference
                # is replaced with the main mention, while all subsequent tokens
                # are masked out with "", so that they can be eliminated from
                # the returned document during "".join(resolved).

                # The first case attempts to correctly handle possessive coreferences
                # by inserting "'s" between the mention and the final whitespace
                # These include my, his, her, their, our, etc.

                # Disclaimer: Grammar errors can occur when the main mention is plural,
                # e.g. "zebras" becomes "zebras's" because this case isn't
                # being explictly checked and handled.

                if final_token.tag_ in ["PRP$", "POS"]:
                    resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                else:
                    # If not possessive, then replace first token with main mention directly
                    resolved[coref[0]] = mention_span.text + final_token.whitespace_
                # Mask out remaining tokens
                for i in range(coref[0] + 1, coref[1] + 1):
                    resolved[i] = ""

        return "".join(resolved)

    def coref_resolved(self, document: str) -> str:
        """
        Produce a document where each coreference is replaced by its main mention

        # Parameters

        document : `str`
            A string representation of a document.

        # Returns

        A string with each coreference replaced by its main mention
        """

        spacy_document = self._spacy(document)
        clusters = self.predict(document).get("clusters")

        # Passing a document with no coreferences returns its original form
        if not clusters:
            return document

        return self.replace_corefs(spacy_document, clusters)

    def _words_list_to_instance(self, words: List[str]) -> Instance:
        """
        Create an instance from words list represent an already tokenized document,
        for skipping tokenization when that information already exist for the user
        """
        spacy_document = Doc(self._spacy.vocab, words=words)
        for pipe in filter(None, self._spacy.pipeline):
            pipe[1](spacy_document)
            
        sentences = []
        
        for sentence in spacy_document.sents:
            for token in sentence:
                sentences.append(token)
        print("done")

        #sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        #sentences = [Token(x) for x in sentences]
        print("done")
        
        instance = self._dataset_reader.text_to_instance(sentences)
        
        return instance

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"document": "string of document text"}`
        """
        document = json_dict["document"]
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance
