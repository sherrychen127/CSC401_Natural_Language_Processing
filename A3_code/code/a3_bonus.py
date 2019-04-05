from a3_gmm import *
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

dataDir = '/u/cs401/A3/data/'
############ work from home
PC = True
if PC:
    dataDir = '/Users/sherrychan/Desktop/CSC401_Assignments/A3_code/data/'
#####################

def train_PCA(d_r):
    testMFCCs = []
    d = 13
    np.random.seed(d)
    X0 = np.empty((0, d))
    pca_array = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print("speaker:", speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)
            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X0 = np.append(X0, myMFCC, axis=0)
    #standardize
    for r in d_r:
        pca = decomposition.PCA(n_components = r)
        pca.fit(X0)
        pca_array.append(pca)
    return pca_array


if __name__ == '__main__':

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8

    scaler = StandardScaler()

    np.random.seed(d)

    d_r = [3]

    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    pca_array = train_PCA(d_r)
    stdout = []
    for a in range(len(pca_array)):
        trainThetas = []
        testMFCCs = []
        pca = pca_array[a]
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print("speaker:", speaker)
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                testMFCCs.append(testMFCC)

                X = np.empty((0, d))
                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X = np.append(X, myMFCC, axis=0)
                X = pca.transform(X)
                trainThetas.append(train(speaker, X, M, epsilon, maxIter))
    # evaluate
        numCorrect = 0
        for i in range(0, len(testMFCCs)):
            testMFCCs_i = pca.transform(testMFCCs[i])
            numCorrect += test(testMFCCs_i, i, trainThetas, k)
        accuracy = 1.0 * numCorrect / len(testMFCCs)

        stdout.append('pca_dim: {} \t M: {} \t maxIter: {} \t Accuracy: {}'.format(d_r[a], M, maxIter, accuracy))
        print("accuracy:", accuracy)
        print('pca_dim: {} \t M: {} \t maxIter: {} \t Accuracy: {}\n'.format(d_r[a], M, maxIter, accuracy))
    #file = open("a3_bonus.txt", "w")
    #file.writelines(stdout)
    #file.close()