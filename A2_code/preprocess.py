import re

#Hansard_train_dir = '/h/u1/cs401/A2_SMT/data/Hansard/Training/'
#Hansard_test_dir = '/h/u1/cs401/A2_SMT/data/Hansard/Testing/'

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function

    '''
    separate sentence-final punctuation, commas, colons, and semicolons, parentheses, dashes between parentheses, mathem
    operators, and quotation marks
    add SENTSTART
    add SENTEND
    '''

    #dash between parenthesis????
    out_sentence = re.sub(r'([?!.,])', r' \1 ', in_sentence)
    out_sentence = re.sub('([:;\(\)\+-<>="])', r' \1 ', out_sentence)
    out_sentence = re.sub('(-)', r' \1 ', out_sentence)

    out_sentence = out_sentence.lower() #lowercase

    if language == 'f': #for french
        out_sentence = re.sub("(l')|(t')|(c')|(j')|(qu')", r"\1\2\3\4\5 ", out_sentence) #l'election -> l' election

        out_sentence = re.sub("(puisqu’on)", "puisqu’ on", out_sentence)
        out_sentence = re.sub("(puisqu’il)", "puisqu’ il", out_sentence)
        out_sentence = re.sub("(lorsqu’il)", "lorsqu’ il", out_sentence)
        out_sentence = re.sub("(lorsqu’on)", "lorsqu’ on", out_sentence)

    #lowercase



    #add SENTSTART and SENTEND
    out_sentence = "SENTSTART" + " " + out_sentence
    out_sentence = re.sub("\n", " SENTEND \n", out_sentence) #insert SENTEND before the \n character
    if '\n' not in out_sentence: #if \n not found in the sentence
        out_sentence = out_sentence + " " + "SENTEND"
    out_sentence = re.sub('\s+', ' ', out_sentence)
    return out_sentence


