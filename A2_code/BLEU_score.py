import math
import string

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
        sentence :	(string) Candidate sentence. "SENTSTART i am hungry SENTEND"
        references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
        n :			(int) one of 1,2,3. N-Gram level.


        OUTPUT:
        bleu_score :	(float) The BLEU score
        """


# TODO: Implement by student.
    if brevity and n == 1: #only return brevity in this case
        can = candidate.split()
        can_len = len(can)
        ref_len = []
        for ref in references:
            ref_list = ref.split()
            ref_len.append(len(ref_list))
        dist = abs(can_len - ref_len[0])
        nearest = ref_len[0]
        for l in ref_len:
            if abs(l - can_len) < dist:
                dist = abs(l-can_len)
                nearest = l
        brev = nearest/can_len
        if brev<1:
            bp = 1
        else:
            bp = math.exp(1-brev)

        return bp

    else:
        #N = number of words n candidate
        can = candidate.strip().split(" ")
        ####################change to list
        ref_string = []
        for i in range(len(references)):
            ref_string.append(references[i].strip().split(" "))

        C = 0
        N = len(can)-(n-1)
        #compute n grams
        for i in range(N):
            token = []
            for j in range(n):
                token.append(can[i+j])

            for r in range(len(ref_string)):
                find = False
                for l in range(len(ref_string[r])- len(token)+1):
                    if ref_string[r][l:l+len(token)] == token:
                        find = True
                if find == True:
                    C += 1
                    break
            #ngram = ' '.join(token) #ngram
            #for reference in references:
            #    if ngram in reference:
            #        C += 1
            #        break #don't double count!
        bleu_score = C/N
        #print(bleu_score)
    return bleu_score



'''
##########test
if __name__ == '__main__':
    candidate = "eng sci rocks"
    references = ["eng sci totally rocks"]
    BLEU_score(candidate, references, 3, False)
    print(BLEU_score(candidate, references, 3, False))
'''