import re

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_word(sentence):
    word_list = []
    sentence = ''.join(sentence.split('  '))
    for i in sentence:
        word_list.append(i)
    return word_list


def get_str(sentence):
    output_str = []
    list = sentence.split('  ')
    for i in range(len(list)):
        if len(list[i]) == 1:
            output_str.append('S')
        elif len(list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(list[i]) - 2
            output_str.append('B')
            output_str.extend('M'* M_num)
            output_str.append('E')
    return output_str

def read_file(filename):
    word, content, label = [], [], []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        letter_list = get_str(eachline)
        word.extend(word_list)
        content.append(word_list)
        label.append(letter_list)
    return word, content, label




