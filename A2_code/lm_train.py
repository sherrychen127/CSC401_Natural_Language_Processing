from preprocess import *
import pickle
import os


def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
	
	# TODO: Implement Function
    language_model = {}
    language_model['uni'] = {}
    language_model['bi'] = {}
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
                    language_model = construct_LM(content, language_model) #use content
    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model




def construct_LM(f, LM):
    for sentence in f:
        tokens = sentence.split()
        #tokens = tokens[1:-1] #exclude SENTSTART and SENTEND
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

def normalize(LM):
    #unigram normalize:
    uni_total = sum(LM['uni'].values())
    for key in LM['uni'].keys():
        LM['uni'][key] = LM['uni'][key] / uni_total

    bi_total = 0
    for key1 in LM['bi'].keys():
        bi_total += sum(LM['bi'][key1].values())
    for key1 in LM['bi'].keys():
        for key2 in LM['bi'][key1].keys():
            LM['bi'][key1][key2] = LM['bi'][key1][key2]/bi_total
    return LM


'''
#comment out
if __name__ == '__main__':
    fn_LM = 'LM_model'
    dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'
    LM = lm_train(dir, 'e', fn_LM)
    #print(LM['bi']['it'])
    #print("unigram:",LM['uni'])
    #print("bigram:", LM['bi'])
'''