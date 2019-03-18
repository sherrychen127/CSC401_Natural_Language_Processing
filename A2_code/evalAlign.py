#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *


#added
from preprocess import * ###comment?

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if not use_cached:
        LM = lm_train(data_dir, language, fn_LM)
    else:
        with open(fn_LM + '.pickle', 'rb') as input_file:
            LM = pickle.load(input_file)
    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached:
        with open(fn_AM + '.pickle', 'rb') as input_file:
            AM = pickle.load(input_file)
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
    return AM

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    BLEU = []
    for i in range(len(eng_decoded)): #one sentence should g
        candidate = eng_decoded[i]
        hansard = preprocess(eng[i],'e') #convert to lowercase
        google = preprocess(google_refs[i], 'e')#convert to lowercase
        references = [hansard, google]
        p = []
        for j in range(1,n+1):
            score = BLEU_score(candidate, references, j, brevity=False)
            p.append(score)
        bp = BLEU_score(candidate, references, 1, brevity= True)

        pn = 1
        for pi in p:
            pn*=pi
        bp_score = bp*pn**(1/n)
        BLEU.append(bp_score)

    return BLEU

def translate(fre_content, LM, AM): #file
    eng_content =[]
    for i in range(len(fre_content)): #sentence
        fre = fre_content[i]
        processed_fre = preprocess(fre, 'f')
        eng = decode.decode(processed_fre, LM, AM)
        eng_content.append(eng)
    return eng_content

def read_file(file):
    with open(file) as f:
        content = f.readlines()
    return content

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """

    data_dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'

    test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f'
    hansard_ref = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e'
    google_ref = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e'

    '''
    ######################################################################
    pc = False
    if pc:
        data_dir = '/Users/sherrychan/Desktop/CSC401_code/data/Hansard/Training'
        test_dir = '/Users/sherrychan/Desktop/CSC401_code/data/Hansard/Testing/Task5.f'
        hansard_ref = '/Users/sherrychan/Desktop/CSC401_code/data/Hansard/Testing/Task5.e'
        google_ref = '/Users/sherrychan/Desktop/CSC401_code/data/Hansard/Testing/Task5.google.e'
    ####################################################################################
    '''



    LME = _getLM(data_dir, 'e', fn_LM = 'LM_model', use_cached=True)
    fre = read_file(test_dir)
    hansard = read_file(hansard_ref)
    google = read_file(google_ref)
    #AM_names = ['AM_model_1k_50']
    AM_names = ['AM_model_1k_50', 'AM_model_10k_50', 'AM_model_15k_50', 'AM_model_30k_50']
    AMs = []
    for AM_name in AM_names:
        AM = _getAM(data_dir, num_sent=1000, max_iter=10, fn_AM=AM_name, use_cached=True)
        AMs.append(AM)

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##
    #print(translate(['estimons'], LME, AMs[1]))
    f = open("Temp.txt", 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")
    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        eng_translation = translate(fre, LME, AM)
        # Eval using 3 N-gram models #
        all_evals = []
        ave_score = []
        for n in range(1, 4):

            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(eng_translation, hansard, google, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            print(AM_names[i],'n:',n, 'BLEU:', evals)
            ave_score.append(sum(evals)/len(evals))
            print("average", sum(evals)/len(evals))
            all_evals.append(evals)
        print("total average:", sum(ave_score)/len(ave_score))
        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)