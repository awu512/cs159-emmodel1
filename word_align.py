import sys
from collections import defaultdict
import time
import os
import psutil
import matplotlib.pyplot as plt

DEFAULT: float = 0.01
UNK_THRESH = 1
OPTIMIZED = True

def read_to_list(path: str) -> set:
    ''' Reads a file into a list of tokenized sentences (sets) '''
    with open(path, 'r') as f:
        return [set(s.split()) for s in f.readlines()]
    
def map_to_ints(sents: set):
    ''' 
    Takes a list of sentence sets and returns equivalent int-mapped senteces and a translation dict.
    Additionally, this function will convert infrequent words (less than 5 occurrences) to '<UNK>'
    '''
    counts = defaultdict(int)
    for s in sents:
        for w in s:
            counts[w] += 1

    i = 1
    word_to_int = { '<UNK>': -1, 'NULL': 0 }
    int_sents = []
    for s in sents:
        int_sent = set()

        for w in s:
            if counts[w] < UNK_THRESH:
                w = '<UNK>'

            # add to the map
            if w not in word_to_int:
                word_to_int[w] = i
                i += 1

            # add int to sentence
            int_sent.add(word_to_int[w])

        int_sents.append(int_sent)

    return int_sents, dict(zip(word_to_int.values(), word_to_int.keys()))


def main(eng_path: str, for_path: str, iters: int, thresh: float):
    ''' 
    Trains on the provided sentence pairs and 
    outputs word pairs and their probablities
    '''

    # init diagnostics
    mem = []
    t = time.time()

    if OPTIMIZED:
        e_sents, e_trans = map_to_ints(read_to_list(eng_path))
        f_sents, f_trans = map_to_ints(read_to_list(for_path))
    else:
        e_sents = read_to_list(eng_path)
        f_sents = read_to_list(for_path)

    # (INT)
    

    probs = {} # { foreign: { english: val }, ... }

    for i in range(iters):
        e_counts: defaultdict[float] = defaultdict(float)
        ef_counts: defaultdict[float] = defaultdict(float) # { (f,e): val, ... }

        for es, fs in zip(e_sents, f_sents):
            # add null word
            if OPTIMIZED:
                es.add(0)
            else:
                es.add('NULL')

            sums = {}

            for e in es:
                for f in fs:
                    
                    # initialize probs on first iter
                    if i == 0: 
                        if f not in probs: probs[f] = {}
                        probs[f][e] = DEFAULT

                    # calculate the sum
                    if f not in sums: sums[f] = sum(probs[f].values())

                    # update the counts
                    f_to_e = probs[f][e] / sums[f]
                    e_counts[e] += f_to_e
                    ef_counts[(f,e)] += f_to_e

            # add this pair's mem usage
            mem.append(psutil.Process(os.getpid()).memory_info().rss)
                
        # recalculate the probabilities
        for (f, e), v in ef_counts.items():
            probs[f][e] = v / e_counts[e]

    # collect, sort, and print the output
    out = []
    for f, e_dict in probs.items():
        for e, v in e_dict.items():
            if v > thresh: 
                if OPTIMIZED:
                    out.append('\t'.join([e_trans[e],f_trans[f],str(v)]))
                else:
                    out.append('\t'.join([e,f,str(v)]))

    out.sort()

    print('\n'.join(out))

    # print/show diagnostics
    # print(f'\nRuntime: {time.time() - t}')
    # plt.plot([i for i in range(len(mem))], mem)
    # plt.show()

if __name__ == '__main__':
    assert len(sys.argv) == 5, f'expected 4 arguments, received {len(sys.argv)-1}'

    eng_path, for_path, iters, thresh = sys.argv[1:5]

    main(eng_path, for_path, int(iters), float(thresh))