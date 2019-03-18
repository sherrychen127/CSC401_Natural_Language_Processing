from preprocess import *
from lm_train import *
from math import log
import string

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

	#TODO: Implement by student.
    #vocabSize = get_word_count(LM)
    #vocabSize = len(LM["uni"])

    #print(vocabSize) #volcab size = 2601242
    tokens = sentence.split()
    #tokens = tokens[1:-1] #exclude SENTSTART and SENTEND
    log_prob = 0
    if smoothing:
        for i in range(len(tokens)-1):
            token1 = tokens[i]
            token2 = tokens[i+1]

            if token1 in LM['bi'].keys() and token2 in LM['bi'][token1].keys():
                bigram = LM['bi'][token1][token2] + delta
            else:
                bigram = delta

            if token1 in LM['uni'].keys():
                unigram = LM['uni'][token1] + delta*vocabSize
            else:
                unigram = delta*vocabSize #delta?

            log_prob += log(bigram/unigram, 2)
    else:
        for i in range(len(tokens)-1):
            token1 = tokens[i]
            token2 = tokens[i+1]

            if token1 in LM['bi'].keys() and token2 in LM['bi'][token1].keys():
                bigram = LM['bi'][token1][token2]
            else:
                bigram = 0

            if token1 in LM['uni'].keys():
                unigram = LM['uni'][token1]
            else:
                unigram = 0
            if unigram == 0 or bigram == 0:
                log_prob = float('-inf')
            else:
                log_prob += log(bigram/unigram, 2)
    #print(log_prob)
    return log_prob

def get_word_count(LM):
    return sum(LM['uni'].values())

def normalize(LM):
    #unigram normalize:
    uni_total = sum(LM['uni'].values())
    #print(uni_total)
    for key in LM['uni'].keys():
        LM['uni'][key] = LM['uni'][key]/uni_total

    bi_total = 0
    for key1 in LM['bi'].keys():
        bi_total += sum(LM['bi'][key1].values())
    for key1 in LM['bi'].keys():
        for key2 in LM['bi'][key1].keys():
            LM['bi'][key1][key2] = LM['bi'][key1][key2]/bi_total
    return LM


'''
if __name__ == '__main__':
    fn_LM = ''
    dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'
    sentence = 'what are you'
    LM = lm_train(dir, 'e', fn_LM)
    print(LM['bi']['word']['.'])
    log_prob(sentence, LM, False)
'''