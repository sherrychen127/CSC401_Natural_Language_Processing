import os
import numpy as np

import fnmatch
dataDir = '/u/cs401/A3/data/'

############ work from home
PC = True
if PC:
    dataDir = '/Users/sherrychan/Desktop/CSC401_Assignments/A3_code/data/'
#####################

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r) # number of words in REF
    m = len(h) # number of words in HYP
    R = np.zeros((n+1, m+1)) #matrix of distance
    B = np.zeros((n+1, m+1))

    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                R[i][j] = max(i, j)
    for i in range(1, n+1):
        for j in range(1, m+1):
            del_n = R[i-1][j] +1
            sub_n = R[i-1][j-1] + (1 - (r[i-1] == h[j-1])) #indicator function: 0 if r == h
            ins_n = R[i][j-1]+1
            R[i][j] = min(del_n, sub_n, ins_n)
            if R[i][j] == del_n:
                B[i][j] = 'up'
            elif R[i][j] == ins_n:
                B[i][j] = 'left'
            else:
                B[i][j] = 'up-left'
    deletion = 0
    insertion = 0
    substitution = 0
    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] == 'up':
                deletion += 1
            elif B[i][j] == 'left':
                insertion += 1
            else:
                substitution += 1
    WER = (deletion + insertion + substitution)/n

    return WER, substitution, insertion, deletion


def preprocess(sentence):
    #remove punctuation except for []
    #lowercase
    


if __name__ == "__main__":
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print("speaker:", speaker)

            ref = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), 'transcripts.txt')
            hyp_google = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), 'transcripts.Google.txt')
            hyp_kaldi = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), 'transcripts.Kaldi.txt')


