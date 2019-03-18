from BLEU_score import *

from preprocess import *
#from evalAlign import *
from perplexity import preplexity
import evalAlign
import pickle
import os


def good_turing_lm(data_dir, language, fn_LM, usercached = True):
    if usercached:
        with open(fn_LM + '.pickle', 'rb') as input_file:
            LM = pickle.load(input_file)
            return LM
    LM = {}
    LM['uni'] = {}
    LM['bi'] = {}
    for subdir, dirs, files in os.walk(data_dir):
        total = len(files)
        for i in range(total):
            file = files[i]
            fullFile = os.path.join(subdir, file)
            #print("processiong:", fullFile, " count:", i+1, '/', total)
            if i%100 == 0 or i == total-1:
                print("processed:", i+1, '/', total)
            if file.endswith(('.'+language)):
                with open(fullFile) as f:
                    f_content = f.readlines()
                    content = [] #preprocessed
                    for sentence in f_content: #preprocess
                        content.append(preprocess(sentence.strip(), language)) #strip
                    LM = construct_GT_LM(content, LM)
    freq = construct_freq(LM)
    LM = GT_smoothing(LM, freq)
    with open(fn_LM + '.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)


def GT_smoothing(LM, freq):
    maximum_uni = max(freq['uni'].values())
    maximum_bi = max(freq['bi'].values())
    for unigram in LM['uni'].keys():
        c = LM['uni'][unigram]
        Nc_1 = find_closest_freq(c, freq, 'uni', maximum_uni)
        Nc = freq['uni'][c]
        LM['uni'][unigram] = (c+1)*Nc_1/Nc
    print("done unigram smoothing")
    for token1 in LM['bi'].keys():
        for token2 in LM['bi'][token1].keys():
            #print(token2)
            c = LM['bi'][token1][token2]
            Nc_1 = find_closest_freq(c, freq, 'bi', maximum_bi)
            Nc = freq['bi'][c]
            LM['bi'][token1][token2] = (c+1)*Nc_1/Nc
    print("done bigram smoothing")
    return LM


def find_closest_freq(count, freq, gram, maximum_freq):
    closest_freq = 10000000
    closest_val = 10000000

    if count == 0:
        for i in range(maximum_freq):
            if i in freq[gram].keys():
                return freq[gram][i]

    for i in range(maximum_freq - count):
        if count + i in freq[gram].keys():
            if i< closest_freq:
                closest_freq = i
                closest_val = freq[gram][count + i]

        if count - i in freq[gram].keys():
            if i < closest_freq:
                closest_freq = i
                closest_val = freq[gram][count - i]
                break


    return closest_val #the count of the ngram with closest frequenct



def construct_GT_LM(f, LM):
    for sentence in f:
        tokens = sentence.split()
        #unigram
        for token in tokens:
            if token in LM['uni'].keys():
                LM['uni'][token] += 1
            else:
                LM['uni'][token] = 1

        #bigram
        for i in range(len(tokens)-1): #exclude sentstart and sent end
            token1 = tokens[i]
            token2 = tokens[i+1]
            if token1 not in LM['bi'].keys():
                LM['bi'][token1] = {}
            if token2 not in LM['bi'][token1].keys():
                LM['bi'][token1][token2] = 1
            else:
                LM['bi'][token1][token2] += 1
    return LM


def construct_freq(LM):
    #unigram
    freq = {}
    freq['uni'] = {}
    freq['bi'] = {}

    for unigram in LM['uni'].keys():
        c = LM['uni'][unigram]
        if c not in freq['uni'].keys():
            freq['uni'][c] = 0
        freq['uni'][c] += 1

    print("done unigram freq")
    #bigram
    count = 0
    for token1 in LM['bi'].keys():
        for token2 in LM['bi'][token1]:
            c = LM['bi'][token1][token2]
            if c not in freq['bi'].keys():
                freq['bi'][c] = 0
            freq['bi'][c] += 1
    print("done freq")
    return freq


if __name__ == '__main__':
    train_dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'
    test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f'
    test_dir_e = '/u/cs401/A2_SMT/data/Hansard/Testing/'
    hansard_ref = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e'
    google_ref = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e'

    AM_name = 'AM_model_15k_50'
    LM = good_turing_lm(train_dir, 'e', 'good_turing_LM', usercached=True)

    AM = evalAlign._getAM(train_dir, num_sent=1000, max_iter=10, fn_AM=AM_name, use_cached=True)

    fre = evalAlign.read_file(test_dir)
    hansard = evalAlign.read_file(hansard_ref)
    google = evalAlign.read_file(google_ref)

    eng_translation = evalAlign.translate(fre, LM, AM)

    all_evals = []
    ave_score = []
    for n in range(1, 4):

        evals = evalAlign._get_BLEU_scores(eng_translation, hansard, google, n)
        print(AM_name, 'n:', n, 'BLEU:', evals)
        ave_score.append(sum(evals) / len(evals))
        print("average", sum(evals) / len(evals))
        all_evals.append(evals)
    print("total average:", sum(ave_score) / len(ave_score))


    print("perplexity is:", preplexity(LM, test_dir_e, 'e', smoothing = False, delta = 0))