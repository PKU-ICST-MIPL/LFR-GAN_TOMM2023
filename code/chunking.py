import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

from flair.data import Sentence
from flair.models import SequenceTagger
from requests.api import get
tagger = SequenceTagger.load("flair/chunk-english")

def get_chunks(sentence):
    entities=[]
    sentence = Sentence(sentence)
    tagger.predict(sentence)
    # print(sentence)

    for entity in sentence.get_spans('np'):
        label=entity.labels[0].value
        # id_text=entity.id_text.split('(')[1].split(')')[0]
        # lst=id_text.split(',')
        # start_pos=int(lst[0])-1
        # end_pos=int(lst[-1])-1+entity.text.count(',')
        entities.append({'start_pos':entity.tokens[0].idx-1,'end_pos':entity.tokens[-1].idx-1,'text':entity.text,'label':label})
    return entities


if __name__ == '__main__':
    # make example sentence
    sent = 'a colorful bird with black wings containing white wingbars, the breast and belly is red.'

    # sentence = Sentence(sent)
    # # predict NER tags
    # tagger.predict(sentence)

    # # print sentence
    # print(sentence)
    # # iterate over entities and print
    # for entity in sentence.get_spans('np'):
    #     print(entity)
    re=get_chunks(sent)
    print(re)
