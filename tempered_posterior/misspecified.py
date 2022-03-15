from functools import lru_cache

import numpy as np
from numpy.random import default_rng
from numpy.linalg import inv, norm, det
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utils import mse, pshape

class Misspecified_Model(object):
    def __init__(self, hparams, priors, params):
        ######### hyper-parameter vars ########
        self.n = hparams.n
        self.A = hparams.A
        self.B = hparams.B
        self.seed = hparams.seed
        
        ######## prior vars ##########
        self.t_mean = priors.t_mean
        self.t_cov = priors.t_cov
        
        self.nu_mean = priors.nu_mean
        self.nu_cov = priors.nu_cov
        
        # inverse of nu and gamma covariance matrices
        self.nu_icov = priors.nu_icov
        
        self.epsilon_mean = priors.epsilon_mean
        self.epsilon_var = priors.epsilon_var
        
        ####### parameter vars ########
        self.alpha = params.alpha

        self.nu = params.nu
        
        self.t = params.t
        self.epsilon = params.epsilon
        self.y = params.y
        
    def init_q_params(self):
        '''
        Initialize the parameters of our approximation density q.
        B/c of our mean field assumption, our covariance matricies aprx_nu_cov and aprx_gamma_cov are only
        defined on their diagonals. We store their diagonal values only in aprx_nu_cov and aprx_gamma_cov.
        '''
                
        self.aprx_nu_mean = np.repeat(np.mean(self.y), self.A)
        self.aprx_nu_cov = np.repeat(np.cov(self.y.T), self.A)
        
    def get_elbo(self):
        '''
        Use hierarchy of variables. We have terms which are made up of expressions 
        which are made of parts. We use the variables ti, ei, and pi respectively for
        the i-ith term, expression, and part
        '''

        e1 = self.y.squeeze() - (self.t @ self.aprx_nu_mean)
        e2 = np.sum(np.power(e1, 2))
        t1 = -0.5 * (self.alpha / self.epsilon_var) * e2

        e4 = np.power(self.aprx_nu_mean - self.nu_mean, 2)
        t2 = -0.5 * (self.nu_icov.T @ e4)
        return t1 + t2
            
    def update_aprx_cov(self):
        # update s nu
        e1 = np.sum(np.power(self.t, 2), axis=0)
        e2 = (self.alpha / self.epsilon_var) * e1
        self.aprx_nu_cov = 1 / (self.nu_icov + e2)
    
    def update_aprx_mean(self):
        #print('Misspecified:\t', self.aprx_nu_mean)
        # copied from update_approx_cov() function
        e1 = np.sum(np.power(self.t, 2), axis=0)
        e2 = (self.alpha / self.epsilon_var) * e1
        t1 = (self.nu_icov + e2)

        # update m nu
        p1 = self.y.squeeze() - (self.t @ self.aprx_nu_mean)
        p2 = np.sum(self.t * p1[:, np.newaxis], axis=0)
        p3 = np.sum(np.power(self.t,2), axis=0) * self.aprx_nu_mean
        e3 = (self.alpha / self.epsilon_var) * (p2 + p3)
        e4 = self.nu_icov * self.nu_mean
        self.aprx_nu_mean = (e3 + e4) / t1
    
    def calc_y_diff(self):
        approx_y = self.generate_approx_data()
        return norm(self.y - approx_y)

    def get_nu_mse(self):
        return mse(self.aprx_nu_mean, self.nu)
    
    def generate_approx_data(self, **kwargs):
        n_input = self.n
        for key in kwargs.keys():
            if key == 'n':
                n_input = kwargs[key]

        # sample nu with variational approximations of mean and cov
        rng = default_rng(self.seed)
        nu_approx = rng.multivariate_normal(self.aprx_nu_mean, np.diag(self.aprx_nu_cov))
      
        # compute y_i = nu*t_i + epsilon_i
        y = (nu_approx[np.newaxis,:] @ np.transpose(self.t[:n_input, :])) + self.epsilon[:n_input]
        return y.T # transpose y to make dimensions work

    def plot_mse(self):
        plt.plot(self.nu_mse)
        plt.legend(['Nu MSE'])
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE of Gamma Approximations (MS)')
        plt.grid(True)
        plt.savefig('plots/MSE_of_Nu_Approximations_(MS).png')
        plt.show()

    def plot_elbo(self):
        plt.plot(self.elbo)
        plt.title('ELBO Values From CAVI Algorithm (S)')
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.grid(True)
        plt.savefig('plots/ELBO_Values_From_CAVI_Algorithm_(S).png')
        plt.show()
    
    def plot_y_diff(self):
        plt.plot(self.y_diff)
        plt.title('Normed Difference Btwn True and Approx Y (S)')
        plt.xlabel('Iteration')
        plt.ylabel('Y Difference')
        plt.grid(True)
        plt.savefig('plots/Normed_Difference_Btwn_True_and_Approx_Y_(S).png')
        plt.show()
    
    def plot_approx_data(self):
        '''
        Only plot a small number of data points b/c otherwise plot gets too messy
        '''
        small_n = 30
        x = np.arange(small_n)
        approx_y = self.generate_approx_data(n=small_n)

        x_coords = np.array([x,x])
        y_coords = np.array([self.y[:small_n], approx_y]).squeeze()
        
        # plot the approx data
        plt.scatter(x, self.y[:small_n], marker='o', facecolors='none', edgecolors='r') # plot true y data pts
        plt.scatter(x, approx_y, marker='s', facecolors='none', edgecolors='g') # plot approx y data pts
        plt.xlabel('Data Point Number')
        plt.ylabel('Y Value')
        plt.grid(True)
        plt.legend(['True Y', 'Approx Y'])
        plt.plot(x_coords, y_coords, color='black') # plot lines btwn true y and approx y
        plt.title('True Y Values and Approximate Y Values (MS)')
        plt.savefig('plots/True_Y_Values_and_Approximate_Y_Values_(MS).png')
        plt.show()

    @lru_cache()
    def fit(self, max_iter=200, tol=1e-8, printing=True):
        self.init_q_params() # initialize q parameters
        self.elbo = [self.get_elbo()] # init and create list to store elbo values
        
        # collect stats about model performance
        self.y_diff = [self.calc_y_diff()] # list to store diff btwn true y and approx y
        self.nu_mse = [self.get_nu_mse()] # mse btwn true nu mean and approx nu mean

        self.update_aprx_cov() # update cov approximation
        for i in range(max_iter):
            if i % 20 == 0 and printing: # print elbo
                print(f'ELBO at {i}:\t{self.elbo[-1]}')
            
            self.update_aprx_mean() # update mean approximation
            self.elbo.append(self.get_elbo()) # calc new EBLO

            # collect stats about model performance
            self.y_diff.append(self.calc_y_diff()) # calc new y diff
            self.nu_mse.append(self.get_nu_mse()) # calc new nu mse

            # check for convergence
            diff = np.abs(self.elbo[-1] - self.elbo[-2])
            if diff < tol:
                if printing:
                    print('ELBO converged with %.3f at iteration %d'%(self.elbo[-1], i+1))
                return

        if printing:
            print('ELBO ended with %.3f after %d iterations'%(self.elbo[-1], max_iter))