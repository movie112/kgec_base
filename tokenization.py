import os
import json
import pandas as pd
from tqdm.auto import tqdm
import sentencepiece as spm

def get_tokenizer(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args.vocab_model_path)
    return sp

# json->txt
def make_txt(json_dir, txt_dir, num):
    json_path = os.path.join(json_dir, f'total_sorted_{num}.json')
    txt_path = os.path.join(txt_dir, f'total_sorted_{num}.txt')

    print(json_path)
    with open(json_path, 'r') as json_file, open(txt_path, 'w') as txt_file:
        raw_dataset = json.load(json_file)
        for item in tqdm(raw_dataset):
            txt_file.write(item[0] + '\n' + item[1])
    return True


# 한 개 
def construct_sp_vocab(input_dir, input_num, prefix, vocab_size, char_coverage, model_type):
    templates = '--input={}\
                 --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}\
                 --bos_id=0 --eos_id=1 --pad_id=2 --unk_id=3\
                 --bos_piece=<s> --eos_piece=</s> --pad_piece=<pad> --unk_piece=<unk>'
    file_path = os.path.join(input_dir, f'total_sorted_{0+9}.txt')
    cmd = templates.format(file_path, prefix, vocab_size, char_coverage, model_type)
    spm.SentencePieceTrainer.train(cmd)

    print('Finished making vocab file.')
    return 
    
if __name__ == '__main__':

    data_dir = '/HDD/kyohoon1/KGEC'
    txt_dir = './data'
    data_num = 30

    prefix = # bpe or char / 저장 위치 및 이름
    model_type = 'char' # bpe or char
    vocab_size = 32000
    char_coverage = 1.0 #0.9999 # char : 1.0

    # make vocab
    if not os.path.exists(prefix + '.model'):
        construct_sp_vocab(txt_dir, data_num+1, prefix, vocab_size, char_coverage, model_type)
    
    print('done')

