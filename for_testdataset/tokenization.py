'''bpe 토큰화할때 보냈다 삭제한거 말고 이거에요!
input_file: noised 문장 전체 txt 파일
prefix: 저장할 경로
vocab_size: 한국학회때 16000,24000,32000 정도로 해봤었는데 간단하게 돌릴대는 작게 두시고 큰 데이터 돌릴때는 삼만 넘어야 할거같아요
char_coverage: 보통 0.9999 썼는데, 결과 너무 좋다 싶으면 0.0001씩 줄여도 돼요 ㅎ.ㅎ'''

import sentencepiece as spm
import json
import pandas as pd
import os

def get_tokenizer(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args.vocab_model)
    return sp


def make_txt(input_file):
    with open(input_file, 'r') as f:
        raw_dataset = json.load(f)
        pairs = [(d['origin'], d['augmented']) for d in raw_dataset['documents']]
        text, labels = list(zip(*pairs))[0], list(zip(*pairs))[1]  

    data = text + labels
    data = pd.DataFrame(data, columns=['text'])

    with open('./data/testdataset.txt', 'w') as f:
        for line in data['text']:
            f.write(line + '\n')
    return True

def construct_sp_vocab(input_file, prefix, vocab_size, char_coverage, model_type):
    # prefix = './tokenizer'
    # vocab_size = 16000
    # char_coverage = 0.9999
    # model_type = 'bpe'

    templates = '--input={}\
                 --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}\
                 --bos_id=0 --eos_id=1 --pad_id=2 --unk_id=3\
                 --bos_piece=<s> --eos_piece=</s> --pad_piece=<pad> --unk_piece=<unk>'
    cmd = templates.format(input_file, prefix, vocab_size, char_coverage, model_type)
    
    spm.SentencePieceTrainer.train(cmd)

    return True

if __name__ == '__main__':
    input_file = './data/testdataset.txt'
    prefix = './tokenizer/char' # bpe or char
    model_type = 'char' #  'char' or 'bpe'
    vocab_size = 16000
    char_coverage = 1.0 # 0.9999
    
    if not os.path.exists(input_file):
        input_json = './data/testdataset.json'
        make_txt(input_json)

    construct_sp_vocab(input_file, prefix, vocab_size, char_coverage, model_type)
    print('done')

    # sp = spm.SentencePieceProcessor()
    # vocab_file = "./tokenizer/bpe.model"
    # sp.load(vocab_file)
    # print(sp.GetPieceSize())
    # lines = [
    # "<s> 나는 그렇게 생각하지 않아..</s><pad>",
    # "나는 이 영화를 보기 위해 오래 기다렸어."
    # ]
    # for line in lines:
    #   print(line)
    #   print(sp.encode_as_pieces(line))
    #   print(sp.encode_as_ids(line))
    #   print()