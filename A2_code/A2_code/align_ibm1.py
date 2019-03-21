from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    #AM = {}

    # Read training data
    (eng, fre) = read_hansard(train_dir, num_sentences)
    # Initialize AM uniformly
    AM = initialize(eng, fre)

    
    # Iterate between E and M steps
    unique_eng, unique_fre = unique_count(eng, fre) #create dictionary of unique eng, fre
    for i in range(max_iter):
        print("EM iteration", i, '/', max_iter)
        AM = em_step(AM, unique_eng, unique_fre)

    #add SENTSTART and SENTEND
    AM['SENTSTART'] = {}
    AM['SENTEND'] = {}
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND']['SENTEND'] = 1

    #store to fn_AM
    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print(AM['0']['0'])
    #print(AM['0']['1'])
    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    eng = []
    fre = []
    for subdir, dirs, files in os.walk(train_dir):
        total = len(files)
        files = sorted(files) #sort the files based on the numbering
        for i in range(total):
            if len(eng) > num_sentences and len(fre) > num_sentences:
                break
            file = files[i]
            print(file)
            fullFile = os.path.join(subdir, file)
            # print("processiong:", fullFile, " count:", i+1, '/', total)
            if i % 100 == 0 or i == total - 1:
                print("processed:", i + 1, '/', total)

            if file.endswith(('.' + 'e')):
                with open(fullFile) as f:
                    e_content = f.readlines()
                    e_lines = read(e_content, 'e')
                    eng.extend(e_lines) #array of processed file
            elif file.endswith(('.' + 'f')):
                with open(fullFile) as f:
                    f_content = f.readlines()
                    f_lines = read(f_content, 'f')
                    fre.extend(f_lines)
    return eng[:num_sentences], fre[:num_sentences]


def read(content, language):
    processed = []
    for i in range(len(content)):
        processed_line = preprocess(content[i].strip(), language) #strip
        #print(processed_line)
        processed.append(processed_line.split(" "))

    return processed





def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    #AM['house']['maison'] = 0.5
    AM = {}

    for i in range(len(eng)):
        eng_sen = eng[i][1:-1] #exclude SENTSTART and SENTEND
        fre_sen = fre[i][1:-1] #exclude SENTSTART and SENTEND
        for eng_word in eng_sen:
            if eng_word not in AM.keys():
                AM[eng_word] = {}
            for fre_word in fre_sen:
                if fre_word not in AM[eng_word].keys():
                    AM[eng_word][fre_word] = 1

    #normalize each e-word entry
    eng_keys = AM.keys()
    for e in eng_keys:
        for f in AM[e].keys():
            AM[e][f] = AM[e][f]/len(AM[e])
    AM['SENTSTART'] = {}
    AM['SENTEND'] = {}
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND']['SENTEND'] = 1
    return AM


def unique_count(eng, fre):
    unique_eng = [] #array of dictionary
    unique_fre = []

    if len(eng) != len(fre):
        print("error, different length of eng and fre")
        return (-1,-1)
    for i in range(len(eng)):
        eng_sen = eng[i][1:-1] #exclude SENTSTART and SENTEND
        fre_sen = fre[i][1:-1] #exclude SENTSTART and SENTEND
        unique_eng.append(unique(eng_sen))
        unique_fre.append(unique(fre_sen))
    return unique_eng, unique_fre



def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    tcount, total = construct_t(eng, fre)
    e_len = len(eng)
    for i in range(e_len):
        eng_count = eng[i] #dictionary
        fre_count = fre[i]
        for f in fre_count.keys():
            denom_c = 0
            for e in eng_count.keys():
                denom_c += t[e][f]*fre_count[f] #denom_c += P(f|e) * F.count(f)
            for e in eng_count.keys():
                add = t[e][f] * fre_count[f] *eng_count[e] / denom_c
                tcount[e][f] += add
                total[e] += add
                #tcount = safe_add_tcount(tcount, e, f, add)
                #total = safe_add_total(total, e, add)
    for e in total.keys():
        for f in tcount[e].keys():
            t[e][f] = tcount[e][f]/total[e]


    return t

def construct_t(eng, fre):
    tcount  = {}
    total = {}
    for i in range(len(eng)):
        eng_d = eng[i]
        fre_d = fre[i]
        for e in eng_d.keys():
            if e not in tcount.keys():
                tcount[e] = {}
            if e not in total.keys():
                total[e] = 0
            for f in fre_d.keys():
                tcount[e][f] = 0
    return tcount, total


def safe_add_tcount(tcount, e, f, add):
    if e not in tcount.keys():
        tcount[e] = {}
    if f not in tcount[e].keys():
        tcount[e][f] = add
    else:
        tcount[e][f] += add
    return tcount


def safe_add_total(total, e, add):
    if e not in total.keys():
        total[e] = add
    else:
        total[e] += add
    return total


def unique(sen):
    ret_sen = {}
    for word in sen:
        if word not in ret_sen.keys():
            ret_sen[word] = 1
        else:
            ret_sen[word] += 1
    return ret_sen



##comment out

if __name__ == '__main__':
    train_dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'

    #####################if running on my own laptop#################
    pc = False
    if pc:
        train_dir = '/Users/sherrychan/Desktop/CSC401_code/data/Hansard/Training'
    #################################################################

    #valid iteration does it converge?
    #align_ibm1(train_dir, 5000, 3, 'AM_model_5k_3')
    #align_ibm1(train_dir, 5000, 10, 'AM_model_5k_10')
    #align_ibm1(train_dir, 5000, 30, 'AM_model_5k_30')
    #align_ibm1(train_dir, 5000, 50, 'AM_model_5k_50')
    #align_ibm1(train_dir, 5000, 100, 'AM_model_5k_100')

    #align_ibm1(train_dir, 1000, 50, 'AM_model_1k_50')

    AM = align_ibm1(train_dir, 1000, 50, 'AM_model_1k_50') #1k done
    #align_ibm1(train_dir, 5000, 10, 'AM_model_5k_10') #5k done
    #align_ibm1(train_dir, 5000, 10, 'AM_model_5k_10') #5k donw
    #align_ibm1(train_dir, 10000, 50, 'AM_model_10k_50') #10k
    #align_ibm1(train_dir, 15000, 50, 'AM_model_15k_50') #15k
    #align_ibm1(train_dir, 30000, 50, 'AM_model_30k_50') #30k
    #align_ibm1(train_dir, 300, 4, 'AM_model_4')
    #align_ibm1(train_dir, 300, 10, 'AM_model_10')

    print(AM['SENTEND'])