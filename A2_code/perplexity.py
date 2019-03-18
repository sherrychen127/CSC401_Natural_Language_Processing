from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp


#test
#train perplexity of the data at testing directory
#use language models learned in Task2
#for each language
#for both MLE and add delta smoothing
#for 3-5 different delta

test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing/'
#english
test_LME = lm_train(test_dir, "e", "eng_test_perplexity_LM")
print("English LM, MLE:", preplexity(test_LME, test_dir, "e", smoothing = False, delta = 0))
print("English LM smoothing, delta = 0.0001:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 0.0001))
print("English LM smoothing, delta = 0.001:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 0.001))
print("English LM smoothing, delta = 0.01:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 0.01))
print("English LM smoothing, delta = 0.1:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 0.1))
print("English LM smoothing, delta = 1:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 1))
print("English LM smoothing, delta = 2:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 2))
print("English LM smoothing, delta = 5:", preplexity(test_LME, test_dir, 'e', smoothing = True, delta = 5))


#french
test_LMF = lm_train(test_dir, 'f', 'fre_test_perplexity_LM')
print("French LM, MLE:", preplexity(test_LMF, test_dir, 'f', smoothing = False, delta = 0))

print("French LM smoothing, delta = 0.0001:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 0.0001))
print("French LM smoothing, delta = 0.001:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 0.001))
print("French LM smoothing, delta = 0.01:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 0.01))
print("French LM smoothing, delta = 0.1:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 0.1))
print("French LM smoothing, delta = 1:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 1))
print("French LM smoothing, delta = 2:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 2))
print("French LM smoothing, delta = 5:", preplexity(test_LMF, test_dir, 'f', smoothing = True, delta = 5))
