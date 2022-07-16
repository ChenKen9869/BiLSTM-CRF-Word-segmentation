import pickle
import lstm_model
import pandas as pd

with open('model/dic.pkl', 'rb') as inp:
    dic = pickle.load(inp)

word_size = 128
maxlen = 500

model = lstm_model.create_model(maxlen, dic, word_size,True)
model.load_weights('model/model.h5', by_name=True)

import re
import numpy as np

tp = {'BE': 0.5,
      'BM': 0.5,
      'EB': 0.5,
      'ES': 0.5,
      'ME': 0.5,
      'MM': 0.5,
      'SB': 0.5,
      'SS': 0.5
      }

tp = {i: np.log(tp[i]) for i in tp.keys()}


def viterbi(nodes):
    paths = {'B': nodes[0]['B'], 'S': nodes[0]['S']}
    for layer in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for node_now in nodes[layer].keys():
            sub_paths = {}
            for path_last in paths_.keys():
                if path_last[-1] + node_now in tp.keys():
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + tp[
                        path_last[-1] + node_now]
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()
            node_subpath = sr_subpaths.index[-1]
            node_value = sr_subpaths[-1]
            paths[node_subpath] = node_value
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()
    return sr_paths.index[-1]


def simple_cut(s):
    if s:
        cutlist = list(s)
        l = 0
        for i in cutlist:
            l = l + 1
            if i not in dic:
                a = ''.join(cutlist[:l - 1])
                b = ''.join(cutlist[l :])
                s1 = simple_cut(''.join(a))
                s2 = simple_cut(''.join(b))
                s1.append(i)
                return s1 + s2

        r = model.predict(np.array([list(dic[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                          verbose=False)[
                0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['S', 'B', 'M', 'E'], i[:4])) for i in r]
        if nodes:
            t = viterbi(nodes)
            words = []
            for i in range(len(s)):
                if t[i] in ['S', 'B']:
                    words.append(s[i])
                else:
                    words[-1] += s[i]
            return words
        else:
            return []
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')

def cutWord(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


