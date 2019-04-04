import os
import numpy as np


import string
import re
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
    #>>> wer("who is there".split(), "is there".split())
    0.333 0 0 1                                                                           
    #>>> wer("who is there".split(), "".split())
    1.0 0 0 3                                                                           
    #>>> wer("".split(), "who is there".split())
    Inf 0 3 0                                                                           
    """
    n = len(r) # number of words in REF
    m = len(h) # number of words in HYP
    R = np.zeros((n+1, m+1)) #matrix of distance
    B = np.zeros((n+1, m+1))

    UP = 1
    LEFT = 2
    UPLEFT = 3
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                R[i][j] = max(i, j)
            if i == 0 and j!= 0: B[i][j] = LEFT
            if j == 0 and i != 0: B[i][j] = UP

    for i in range(1, n+1):
        for j in range(1, m+1):
            del_n = R[i-1][j] +1
            sub_n = R[i-1][j-1] + (1 - (r[i-1] == h[j-1])) #indicator function: 0 if r == h
            ins_n = R[i][j-1]+1
            R[i][j] = min(del_n, sub_n, ins_n)
            if R[i][j] == del_n:
                B[i][j] = UP
            elif R[i][j] == ins_n:
                B[i][j] = LEFT
            else:
                B[i][j] = UPLEFT

    substitution, insertion, deletion = backtrack(R, B, n, m)
    WER = R[n][m]/n
    return WER, substitution, insertion, deletion

def backtrack(R, B, n, m):
    UP = 1
    LEFT = 2
    UPLEFT = 3

    deletion = 0
    insertion = 0
    substitution = 0
    while n>0 or m>0:
        cur = R[n][m]
        if B[n][m] == UP:
            n = n-1
            prev = R[n][m]
            if cur!= next: deletion += cur - prev
        elif B[n][m] == LEFT:
            m = m-1
            prev = R[n][m]
            if cur!= next: insertion += cur - prev
        elif B[n][m] == UPLEFT:
            m = m-1
            n = n-1
            prev = R[n][m]
            if cur!= next: substitution += cur - prev
    return int(substitution), int(insertion), int(deletion)


def preprocess(sentence):
    sentence = re.sub('<[^>]*>', '', sentence)     #remove label
    for c in string.punctuation:
        if c != '[' and c != ']':
            sentence = sentence.replace(c, " ")    #remove punctuation except for []
    sentence = sentence.lower()    #lowercase
    sentence = re.sub('\s+', ' ', sentence) #remove the multiple spaces
    return sentence.split()



if __name__ == "__main__":
    #test
    #sentence = "0 LD/E:INTERACTIVE  m- / so so on it./ I mean, <BR> I don't know./ <BR> I did alright./ <LG> "
    #sentence = preprocess(sentence)
    #print(Levenshtein("".split(), "who is there".split()))
    google_wer_history = []
    kalbi_wer_history = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print("speaker:", speaker)

            ref_dir = os.path.join( dataDir, speaker, 'transcripts.txt')
            hyp_google_dir = os.path.join( dataDir, speaker, 'transcripts.Google.txt')
            hyp_kaldi_dir =  os.path.join( dataDir, speaker, 'transcripts.Kaldi.txt')

            ref_file = open(ref_dir, 'r')
            google_file = open(hyp_google_dir, 'r')
            kalbi_file = open(hyp_kaldi_dir, 'r')

            ref = ref_file.readlines()
            google = google_file.readlines()
            kalbi = kalbi_file.readlines()

            #print(len(ref), len(google), len(kalbi))
            #skip speakers when the transcript file is empty
            if len(ref)>0:
                for i in range(len(ref)):
                    #preprocess
                    ref_sentence = preprocess(ref[i])
                    google_sentence = preprocess(google[i])
                    kalbi_sentence = preprocess(kalbi[i])
                    wer_google = Levenshtein(ref_sentence, google_sentence)
                    wer_kalbi = Levenshtein(ref_sentence, kalbi_sentence)

                    google_wer_history.append(wer_google)
                    kalbi_wer_history.append(wer_kalbi)
                    print("{} {} {} {} S:{} I:{} D:{}".format(speaker, "google", i, wer_google[0], wer_google[1], wer_google[2], wer_google[3]))
                    print("{} {} {} {} S:{} I:{} D:{}".format(speaker, "kalbi", i, wer_kalbi[0], wer_kalbi[1], wer_kalbi[2], wer_kalbi[3]))



                    #average and standard deviation

                    #[SPEAKER] [SYSTEM] [i] [WER] S:[numSubstitutions], I:[numInsertions], D:[numDeletions]
            ref_file.close()
            google_file.close()
            kalbi_file.close()
    print("google average:{}, std: {}".format(np.average(google_wer_history), np.std(google_wer_history)))
    print("kalbi average:{}, std: {}".format(np.average(kalbi_wer_history), np.std(kalbi_wer_history)))



