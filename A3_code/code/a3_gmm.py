from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random


from math import *
from scipy.special import logsumexp
import sys
dataDir = '/u/cs401/A3/data/'


############ work from home
PC = True
if PC:
    dataDir = '/Users/sherrychan/Desktop/CSC401_Assignments/A3_code/data/'
#####################

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''

    mu_m = myTheta.mu[m] #1
    cov_m = myTheta.Sigma[m] #(1,d)

    term1 = -1*np.sum(0.5*np.square(x)/cov_m - mu_m*x/cov_m)

    if len(preComputedForM) > 0:
        log_b_m = term1 - preComputedForM[m]
        return log_b_m
    d = len(x)
    #term2 = 0.5*np.sum(np.square(mu_m)/cov_m) + d/2*log(2*np.pi) + 0.5*np.log(np.prod(cov_m))
    term2 = 0.5 * np.sum(np.square(mu_m)/cov_m) + d/2*np.log(2*np.pi) + 1/2 * np.sum(np.log(myTheta.Sigma[m]))
    log_b_m = term1 - term2
    '''
    x_minus_mu = x-mu_m #(1,d)
    exponent = np.square(x_minus_mu)/cov_m #1/[1,d]
    exponent = -0.5*np.sum(exponent)
    d = len(x)
    denominator = d/2*log(2*pi) + 1/2*np.sum(np.log(cov_m))
    log_b= exponent - denominator
    '''
    return log_b_m






def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    weights = myTheta.omega
    w = weights[m]
    log_b = log_b_m_x(m, x, myTheta) #log == ln????
    log_w_b = scipy.special.logsumexp(log_b, b = w)

    M = np.shape(weights)[0]
    log_bk = np.array([log_b_m_x(i, x, myTheta) for i in range(M)])
    log_wk_bk = scipy.special.logsumexp(log_bk, b = weights)
    log_p_m_x_val = log_w_b - log_wk_bk
    return log_p_m_x_val

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    log_w = np.log(myTheta.omega)
    log_wm_bm = np.sum(logsumexp(log_Bs + log_w, axis = 0))
    return log_wm_bm


def log_b_x(X, myTheta, preComputedForM=[]):
    '''
    :param M:
    :param X: (T, d)
    :param myTheta:
    :param preComputedForM:
    :return: log_b_x, (M,T)
    '''

    T, D = np.shape(X)
    mu = myTheta.mu #(M, d)
    cov = myTheta.Sigma #(M, d)
    X_squared = np.reshape(np.trace(np.dot(X, X.T)), (T, 1)) #(T, 1)
    term1 = -1*np.sum(0.5*X_squared/cov[:,None,:] - np.dot(mu, X.T)[:,:,None]/cov[:,None,:]) ###not sure about the broach case dimension
    if len(preComputedForM) > 0:
        preComputedForM = np.array(preComputedForM)
        log_b = term1 - preComputedForM
        return log_b
    preComputedForM = pre_computed_for_M(X, myTheta)
    log_b = term1 - preComputedForM
    return log_b

def log_p_x(log_b, myTheta):
    w = myTheta.omega.ravel() #(M, ) 1D array
    log_w_b = np.transpose(np.log(w) + np.transpose(log_b))
    log_wk_bk = np.sum(log_w_b, axis = 0)
    log_p = log_w_b - log_wk_bk
    return log_p

def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    ##initialization
    myTheta.omega = np.zeros((M,1)) + 1/M #initialize weights uniformly
    rand_ind = np.random.randint(X.shape[0], size= M)
    myTheta.mu = X[rand_ind,:]# (M,d)
    myTheta.Sigma = np.ones((M, d))
    i = 0
    prev_L = float('-inf')
    improvement = float('inf')
    while i<= maxIter and improvement > epsilon:
        #compute Intermediate Result
        #log_b_x, log_p_x = calculate_intermediate_result(X, M, myTheta)
        log_Bs = compute_log_Bs(myTheta, X)
        log_WB = log_Bs + np.log(myTheta.omega)
        log_P = log_WB - logsumexp(log_WB, axis=0)

        L = logLik(log_Bs, myTheta) #compute log-likelihood

        myTheta = update_parameter(log_P, X, myTheta) #update parameter
        improvement = L - prev_L
        prev_L = L
        i += 1
        print("log_likelihood:", L)
    return myTheta


def update_parameter(log_p_x, X, myTheta):
    p_x = np.exp(log_p_x) # M, T
    M, T = np.shape(p_x)
    for m in range(M):
        p_m_x = p_x[m]
        p_sum_m = np.sum(p_m_x)
        #update weights
        myTheta.omega[m] = p_sum_m/T
        #mu
        myTheta.mu[m] = np.dot(p_m_x, X)/p_sum_m
        #sigma
        sigma_m = (np.dot(p_m_x, np.square(X))/p_sum_m) - (np.square(myTheta.mu[m]))
        myTheta.Sigma[m] = sigma_m
    '''
    sum_p_x = np.reshape(np.sum(p_x, axis=1),(M, 1)) #(M, 1)
    myTheta.omega = sum_p_x/T # (M, 1) update weights

    #update variance (variance first or mean first?)
    X_squared = np.square(X) # X^2 = (T, d)
    sum_p_x_squared = np.dot(p_x, X_squared) #(M, d)
    mu_squared = np.square(myTheta.mu)
    E_X_squared = sum_p_x_squared / sum_p_x #(M, d)
    var = E_X_squared - mu_squared # (M, d)
    myTheta.Sigma = var

    #update mean
    myTheta.mu = np.dot(p_x, X)/sum_p_x  #( M,d)/(M,1) = (M, d), update mean
    '''
    return myTheta


def pre_computed_for_M(X, myTheta):
    (T, d) = np.shape(X)
    mean = myTheta.mu #(M,1)
    cov = myTheta.Sigma #(M, d)

    cov_recip = np.reciprocal(cov) #calculate reciprocal
    term1 = 0.5*np.sum(np.multiply(np.square(mean), cov_recip), axis= 1) #is it (M, d)-> (M,1)
    term2 = d/2*log(2*pi)
    term3 = 0.5*np.log(np.prod(cov, axis = 1)) #(M,1)
    preComputedForM = term1 + term2 + term3
    return preComputedForM


def calculate_intermediate_result(X, M, myTheta):
    PreComputedForM_val = pre_computed_for_M(X, myTheta)
    log_bx = [[log_b_m_x(m, X[i], myTheta, PreComputedForM_val) for i in range(X.shape[0])] for m in range(M)] #(M, T)
    #log_bx = log_b_x(X, myTheta, preComputedForM = PreComputedForM_val)
    print("done log_bx")
    #log_px = [[log_p_m_x(m, X[i], myTheta) for i in range(X.shape[0])] for m in range(M)] #(M,T)
    log_px = log_p_x(log_bx, myTheta)
    print("done log_p_x")
    return log_bx, log_px


def compute_log_Bs(myTheta, X):
    T, d = np.shape(X)
    M, a = np.shape(myTheta.omega)
    Log_Bs = np.zeros((M, T))
    for m in range(M):
        term1 = np.sum(np.square(X - myTheta.mu[m]) / myTheta.Sigma[m], axis=1)
        term2 = d/2 * np.log(2*np.pi)
        term3 = 1./2 * np.sum(np.log(np.square(np.prod(myTheta.Sigma[m]))))
        log_bs = - term1 - term2 - term3
        Log_Bs[m] = log_bs
    return Log_Bs



def test(mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''

    bestModel = -1
    log_like = []
    for model in models: #(30 models/speakers, theta)
        log_Bs = compute_log_Bs(model, mfcc)
        log_like.append(logLik(log_Bs, model))

    bestModel = np.argmax(np.array(log_like))
    #print("correctID:", correctID)
    #print("bestmodel:", bestModel)
    print(models[correctID].name)
    if k>0:
        k_best_models = np.argsort(np.array(log_like))[-k:][::-1] #reverse order

        #print("k_best_model:", k_best_models)
        for i in k_best_models:
            print(models[i].name, ' ', log_like[i])
    ###########write to stdout#########################




    ###################################################
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( "speaker:", speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)
            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("accuracy:", accuracy)


