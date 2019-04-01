from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from math import *


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

    mu_m = myTheta.mu[m]
    cov_m = np.diag(myTheta.Sigma[m])

    x_minus_mu = x-mu_m

    exponent = np.dot(x_minus_mu.T, x_minus_mu)/cov_m

    exponent = -0.5*np.sum(exponent)
    denominator = d/2*log(2*pi) + 1/2*np.sum(np.log(cov_m))

    #log_b_per_t = -0.5*(exponent + np.log(2* pi * cov_m))
    #log_b = sum(log_b_per_t, 2)

    log_b= exponent - denominator
    return log_b
    
def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    weights = myTheta.weights
    w = weights[m]
    log_b = log_b_m_x(m,x, myTheta)
    b = np.exp(log_b)
    w_b = b*w
    bk = np.array([log_b_m_x(i,x,myTheta) for i in range(np.size(weights)[0])])
    wk_bk = np.dot(weights.T, bk.reshape(np.size(bk)[0], 1))
    p_m_x = w_b/wk_bk

    #sum_w_b = np.sum(w_b)
    #p_m_x = w_b/sum_w_b

    return p_m_x

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    B_s = np.exp(log_Bs)
    temp = np.dot(B_s.T, myTheta.omega)
    res = np.sum(temp)
    return res

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    ##initialization
    myTheta.omega = np.zeros((M,1)) + 1/M #initialize weights uniformly
    rand_ind = np.random.randint(X.shape[0], size= M)
    myTheta.mu = X[rand_ind,:]# (M,d)
    myTheta.Sigma = np.ones((M,d))



    i = 0
    prev_L = float('-inf')
    improvement = float('inf')
    while i<= maxIter and improvement >= epsilon:
        #compute Intermediate Result
        log_b_x, log_p_x = calculate_intermediate_result(X, M, myTheta)
        L = logLik(log_b_x, myTheta) #compute log-likelihood
        myTheta = update_parameter(log_b_x, log_p_x, X, myTheta) #update parameter
        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def update_parameter(log_b_x, log_p_x, X, myTheta):
    p_x = np.exp(log_p_x) # M, T

    myTheta.omega = np.sum(p_x, axis = 1)/p_x.shape[1] # (M, 1)
    myTheta.mu = np.dot(p_x, X)/np.sum(p_x, axis = 1) #(M, d)

    X_2 = X.dot(X) # X^2 = (T,D)
    sum_p_x_2 = X_2.T.dot(p_x)

    mu_2 = np.dot(myTheta.mu, myTheta.mu)
    E_X_squared = sum_p_x_2 / np.sum(p_x, axis = 1)
    var = E_X_squared - mu_2 #D, M
    myTheta.Sigma = var.T

    return myTheta



def calculate_intermediate_result(X, M, myTheta):
    log_b_x = [[log_b_m_x(m, X[i], myTheta) for i in range(X.shape[0])] for m in range(M)] #(M.T)
    #b_x = np.exp(log_b_x) #(M.T)

    log_p_x = [[log_p_m_x(m, X[i], myTheta) for i in range(X.shape[0])] for m in range(M)] #(M,T)
    #p_x = np.exp(log_p_x)#(M,T)

    return log_b_x, log_p_x




def test( mfcc, correctID, models, k=5 ):
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
        log_b_x, log_p_x = calculate_intermediate_result(mfcc, model.omega.shape[0], model)
        log_like.append(logLik(log_b_x, model))
    bestModel = np.argmax(log_like)

    ###########write to stdout#########################




    ###################################################
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    '''
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
            print( speaker )

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
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)
    '''

