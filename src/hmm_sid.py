import numpy as np
import pathlib
import time
import librosa
import os
import warnings
import argparse
import copy
import tqdm
from mfcc import _WindowingMFCC
from delta import compute_delta
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
import pdb
import json


def multivariate_normal_logpdf(x, mean, covar):
    """
    This function calculates the log of the probability density function of a multivariate normal distribution.
    """
    x_diff = x - mean
    # breakpoint()
    # breakpoint()
    try:
        inv_covar = np.linalg.inv(covar + 1e-10)
    except np.linalg.LinAlgError: 
        breakpoint()
    log_det_covar = np.linalg.slogdet(covar)[1]
    try:
        quadratic_term = -0.5 * np.dot(x_diff.T, np.dot(inv_covar, x_diff))
    except ValueError :
        breakpoint()
    normalization_term = -0.5 * (len(x) * np.log(2 * np.pi) + log_det_covar)
    # breakpoint()
    return quadratic_term + normalization_term







class HMMGaus():
    def __init__(self, n_states, A, mu, sigma, pi):
        self.n_states = n_states
        self.A = A
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def log_sum_exp(self, log_probs):
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
    
    def Forward(self, observation):
        '''
        alphas should be in log domain to avoid underflow and overflow
        assume that every other thing is in log domain
        '''
        self.T = observation.shape[0]
        #initialize empty or zero alphas 
        #observations are of shape (T, 26)
        alpha = np.full((self.n_states, self.T), -np.inf)
        
        #calculating the emission_probabilities
        b = np.full((self.n_states,self.T ), -np.inf)

        for states in range(self.n_states):
            for times in range(self.T):
                #print("shape of mu passed is ",self.mu[states].shape)
                #print("shape of sigmas passed is ", self.sigma[states].shape)
                b[states, times] = multivariate_normal_logpdf(x = observation[times, :], mean= self.mu[states], covar=self.sigma[states])
        self.b = np.asarray(b)
    
        #calculating the alpha_t=1 (i)
        alpha[:, 0] = self.pi + self.b[:, 0]  #pi_{state = i} * b_{state = i} (O_{1})

        #calculating the alpha_t=2:T(i)

        for t in range(1, self.T):
            for i in range(self.n_states):
                alpha[i, t] = self.b[i, t] + self.log_sum_exp(alpha[:, t-1] + self.A[:, i])
        
        #--update : was not included
        # p_o_lambda = -1e30
        # for j in range(self.n_states):
        #     p_o_lambda = np.logaddexp(p_o_lambda, alpha[j, -1])
        
        self.alpha = alpha
        return alpha
        # self.p_o_lambda = p_o_lambda
        # return self.alpha
    
    def Backward(self):
        '''
        betas should be in log domain to avoid underflow and overflow
        assume that every other thing is in log domain
        '''

        #initializing empty beta array
        #observations are of shape (T, 26)
        beta = np.full((self.n_states, self.T), -np.inf) 

        #calculating the initial condition of beta
        beta[:, self.T-1] = 0 #beta_{t=T}(i) = 1

        for time in range(self.T-2, -1, -1):
            for state in range(self.n_states):
                beta[state, time] = self.log_sum_exp(self.A[state,:] + self.b[:, time+1] + beta[:, time+1])
    

        self.beta = beta
        # return self.beta
    
    def GammaandXi(self):
        #self.p_o_lambda was not included
        #gamma = self.alpha + self.beta - self.p_o_lambda

        #updated
        gamma = self.alpha + self.beta

        max_gamma = np.max(gamma, axis = 0, keepdims = True) #along each states

        gamma-= max_gamma

        gamma_norm = gamma - self.log_sum_exp(gamma)

        self.gamma = gamma_norm

        xi = np.full((self.n_states, self.n_states, self.T -1), -np.inf)
        for t in range(self.T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[i,j,t] = self.alpha[i, t] + self.A[i, j] + self.b[j, t+1] + self.beta[j, t+1] 
            xi[:, :, t] -=self.log_sum_exp(xi[:, :, t])
        
        self.xi = xi

        # return gamma, xi


    def EM(self, observations, thresh=0.000001, maxIter=300):
        log_likelihood = []
        iter = 0
        while iter < maxIter:

            log_likelihood_iter = []

            for observation in observations:
                # E step
                # get forward, backward, gamma and xi
                self.Forward(observation=observation)
                self.Backward()
                self.GammaandXi()
                # print("Alpha is \n", self.alpha)
                # print("Beta is \n", self.beta)
                # print("Gamma is \n", self.gamma)
                # print("Xi is \n", self.xi)
                # breakpoint()
                # M step
                # updating pi
                self.pi = self.gamma[:, 0] - self.log_sum_exp(self.gamma[:, 0])

                # print("pi is \n", self.pi)



                # re-estimating A
                tp_A = np.full((self.n_states,self.n_states), -np.inf)
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        tp_A[i, j] = self.log_sum_exp(self.xi[i, j, :]) - self.log_sum_exp(self.gamma[i, :-1])
                
                self.A = tp_A
                # print("A is \n", self.A)    

                # re-estimating mean and covariance for b
                
                lg_means = np.full(self.mu.shape, -np.inf) #(n_states, 26)
                lg_cov = np.full(self.sigma.shape, -np.inf) #(n_states, 26, 26)
                eps = 1e-7
                for i in range(self.n_states):
                    sum_gamma = self.log_sum_exp(self.gamma[i, :]) #constant
                    for dims in range(26):
                        wt_sum = self.log_sum_exp(self.gamma[i, :] + np.log(observation[:, dims] + eps))
                        lg_means[i, dims] = wt_sum - sum_gamma
                    # breakpoint()
                    for t in range(self.T):
                        diff = np.log(observation[t] + eps) - lg_means #(n_ states, 26)
                        wt_diff = self.gamma[i, t] + diff - sum_gamma #(n_ states, 26)

                        for dim1 in range(26):
                            for dim2 in range(26):
                                curr_cov = lg_cov[i, dim1, dim2]
                                cov_updated = wt_diff[i, dim1] + diff[i, dim2] #(10, )
                            

                               
                                lg_cov[i, dim1, dim2] = self.log_sum_exp(np.array([curr_cov, cov_updated]))
                    
                    for di in range(26):
                        lg_cov[i, di, di] = self.log_sum_exp(np.array([lg_cov[i,di,di], np.log(eps)]))
                # breakpoint()
                self.mu = np.exp(lg_means)
                self.sigma = np.exp(lg_cov)

                # print("Means is \n", self.mu)
                # print("Covar is \n", self.sigma)
                #computing the likelihood
                

                log_likelihood_iter.append(self.log_sum_exp(self.alpha[:, -1]))


            log_likelihood.append(np.mean(log_likelihood_iter))
            print(f"Iter number:{iter} and log_likelihood : {log_likelihood[-1]}")
            if len(log_likelihood) > 2 and abs(log_likelihood[-1] - log_likelihood[-2]) < thresh:
                print("Converged")
                return self.pi, self.A, self.mu, self.sigma, log_likelihood
            iter += 1

        return self.pi, self.A, self.mu, self.sigma, log_likelihood


def test(audio, sr):
        y = audio
        mfcc_features = []
        # breakpoint()
        # mfcc_features.append(compute_delta()._forward(y, sr).T)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T 
        # breakpoint()
        mfccs = MinMaxScaler().fit_transform(X = mfccs)
        mfcc_features.append(mfccs)

        mfcc_features = np.array(mfcc_features)
        # breakpoint()
        # breakpoint()
        likelihood = {"odessa" : 0, "playmusic": 0 , "turnonlights": 0, "turnofflights": 0, "stopmusic": 0, "whattimeisit": 0}
        for classes in ['odessa','playmusic','turnonlights','turnofflights','whattimeisit','stopmusic']:
            
            with open(f"/Users/sidharth/Desktop/speech_processing/models/{classes}_hmmlearn_fold_1.txt", 'r') as file:
                loaded_list = json.load(file)
            
            pi_1 = np.array(loaded_list['pi'])
            tp_1 = np.array(loaded_list['tp'])
            mean_1 = np.array(loaded_list['mu'])
            sigma_1 = np.array(loaded_list['sigma'])
            # breakpoint()
            hmm = HMMGaus(n_states=10, A=tp_1, mu=mean_1, sigma=sigma_1, pi=pi_1)
            alpha = hmm.Forward(mfcc_features[0])

            log_likelihood = hmm.log_sum_exp(alpha[:, -1])

            likelihood[classes] = log_likelihood

        # print("predicted class is ", max(likelihood, key=lambda k: likelihood[k]) )
        return max(likelihood, key=lambda k: likelihood[k])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help="task in (train, test)}")
    args = parser.parse_args()
    task = args.task
    # task = 'test'
    if task =='train':
        for classes in ['odessa','playmusic', 'stopmusic', 'turnofflights', 'turnonlights', 'whattimeisit']:

            print(f"Training class {classes} .......")

            audio_path_dir = f"/Users/sidharth/Desktop/speech_processing/audio/data/train/{classes}"
            dir = pathlib.Path(audio_path_dir)
            list_audio_files = dir.glob('**/*.wav')
            
            # Extract MFCC features using librosa
            mfcc_features = []
            for audios in list_audio_files:
                y, sr = librosa.load(audios)
                # mfcc_features.append(compute_delta()._forward(y, sr).T)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T 
                # breakpoint()
                mfccs = MinMaxScaler().fit_transform(X = mfccs)
                mfcc_features.append(mfccs)

            # breakpoint()
            mfcc_features = np.asarray(mfcc_features) #((samples, T, dim))
            # mfcc_features = np.expand_dims(mfcc_features[0], 0)

            # Initialize the HMM parameters
            n_states = 10 # Number of hidden states
            transmission_prob = np.full((n_states, n_states), -np.log(n_states))
            pi = np.full(n_states, -np.log(n_states))
            # flattened_features = mfcc_features.reshape(-1, mfcc_features.shape[-1])

            # Calculate global mean for each MFCC feature

            global_mean = np.mean(np.concatenate(mfcc_features, axis=0), axis=0)
            global_variance = np.var(np.concatenate(mfcc_features, axis=0), axis=0)
            # breakpoint()
            # global_mean = np.mean(mfcc_features.reshape(-1, 26),axis=0)

            # Calculate global variance for each MFCC feature
            # global_variance = np.var(mfcc_features.reshape(-1, 26),axis=0)
            # Print global mean and variance
            print("Global Mean:")
            print(global_mean)
            print("\nGlobal Variance:")
            print(global_variance)
            # breakpoint()
            emission_means = global_mean + np.asarray([np.random.randn(26) * 0.125 * global_variance for _ in range(n_states)])
            # breakpoint()
            # Initialize covariance matrices
            emission_covars = np.asarray([np.diag(global_variance) for _ in range(n_states)])
            # breakpoint()

                # return xi_values
            audio_path = f"/Users/sidharth/Desktop/speech_processing/audio/data/train/{classes}"

            # Extract MFCC features using librosa
            print("computing mfcc features...")
            mfcc_features = []
            for audios in os.listdir(audio_path):
                if audios.endswith('.wav'):
                    y, sr = librosa.load(f"{audio_path}/{audios}")
                    # mfcc_features.append(compute_delta()._forward(y, sr).T)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T #T, 26
                    mfccs = MinMaxScaler().fit_transform(X = mfccs)
                    mfcc_features.append(mfccs)



            mfcc_features = np.asarray(mfcc_features) #((samples, T, dim))
            
            

            '''self.n_states = n_states
            self.A = A
            self.mu = mu
            self.sigma = sigma
            self.pi = pi'''
            print("starting hmm computation...")
            hmm = HMMGaus(n_states=n_states, A=transmission_prob, mu=emission_means, sigma=emission_covars, pi=pi)
            pi, tp, mu, sigma, log_likelihood = hmm.EM(observations=mfcc_features)
            model_state_dict = {'pi': pi.tolist(), 'tp': tp.tolist(), 'mu' : mu.tolist(), 'sigma' : sigma.tolist()}
            # breakpoint()
            if os.path.exists(f"/Users/sidharth/Desktop/speech_processing/{classes}_hmmlearn.txt")==True:
                with open(f"/Users/sidharth/Desktop/speech_processing/{classes}_hmmlearn.txt", 'w') as file:
                    
                    json.dump(model_state_dict, file)

    if task=='test':
        for classes in ['odessa','playmusic','turnonlights','turnofflights','whattimeisit','stopmusic']:
            # classes = 'Odessa'
            print("Actual class", classes)
            audio_path_dir = f"/Users/sidharth/Desktop/speech_processing/audio/data/train/{classes}/{classes}_1.wav"
            mfcc_features = []
            y, sr = librosa.load(audio_path_dir)
            # breakpoint()
            # mfcc_features.append(compute_delta()._forward(y, sr).T)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T 
            # breakpoint()
            mfccs = MinMaxScaler().fit_transform(X = mfccs)
            mfcc_features.append(mfccs)

            mfcc_features = np.array(mfcc_features)
            # breakpoint()
            # breakpoint()
            starttime=  time.time()
            likelihood = {"odessa" : 0, "playmusic": 0 , "turnonlights": 0, "turnofflights": 0, "stopmusic": 0, "whattimeisit": 0}
            for classes in ['odessa','playmusic','turnonlights','turnofflights','whattimeisit','stopmusic']:

                with open(f"/Users/sidharth/Desktop/speech_processing/models/{classes}.txt", 'r') as file:
                    loaded_list = json.load(file)
                
                pi_1 = np.array(loaded_list['pi'])
                tp_1 = np.array(loaded_list['tp'])
                mean_1 = np.array(loaded_list['mu'])
                sigma_1 = np.array(loaded_list['sigma'])
                # breakpoint()
                hmm = HMMGaus(n_states=10, A=tp_1, mu=mean_1, sigma=sigma_1, pi=pi_1)
                alpha = hmm.Forward(mfcc_features[0])

                log_likelihood = hmm.log_sum_exp(alpha[:, -1])

                likelihood[classes] = log_likelihood

            print("predicted class is ", max(likelihood, key=lambda k: likelihood[k]) )













            















            