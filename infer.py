from allennlp.predictors.predictor import Predictor 
from allennlp.models import load_archive, archive_model
from allennlp.models import Model
from stary_dataloader import StaryEvalReader, StaryReader
from staryPredictor import CorefPredictorStary
import spacy
import os
import json
import jsonlines


def model_fn(model_dir):
    
    #model_file = os.path.join(model_dir, 'model.tar.gz')
    archive = load_archive('model.tar.gz')
    predictor = Predictor.from_archive(archive, 'core')
    
    return predictor


def input_fn(request_body, request_content_type):
    
    return request_body
    
def predict_fn(input_data, predictor):
    
    outputs = predictor.predict(input_data)
    out = {}
    out['doc'] = ' '.join(outputs['document'])
    out['clusters'] = outputs['clusters']
    
    for cluster in outputs['clusters']:
        print(get_span_words(cluster[0], outputs['document']) + ': ', end='')
        print(f"[{'; '.join([get_span_words(span, outputs['document']) for span in cluster])}]")

    return out

def get_span_words(span, document):
        return ' '.join(document[span[0]:span[1]+1])
    
#def output_fn(prediction, content_type):
    #return prediction


if __name__ == "__main__":
    texts = "After all , he had been asking for her hand in marriage for so many years . Kai was a known cold - hearted brute , and she was surprised he was interested in her at all . "

    predictor = model_fn('../')
    result = predict_fn(texts, predictor)
    print(result)







