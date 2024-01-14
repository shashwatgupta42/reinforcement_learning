import numpy as np
import math
import random as rd

class Deep_Q_Learner:
    def __init__(self, state_size, nodes, activations):
        self.state_size = state_size
        self.nodes = nodes
        self.activations = activations
        self.layers = len(nodes)
        self.actions = nodes[-1]
        self.init_original_param = self.initialise_param()
        self.init_target_param = self.init_original_param.copy()
        self.initialise_adam_cache()
        self.train_original_param = self.init_original_param.copy()
        self.train_target_param = self.init_target_param.copy()

    def initialise_param(self):
        num_nodes = [self.state_size]
        num_nodes.extend(self.nodes)
        total_param = 0
        init_param = {}
        for i in range(1, self.layers+1):
            W = np.random.randn(num_nodes[i], num_nodes[i-1])*0.01
            b = np.zeros((num_nodes[i], 1))
            init_param['W'+str(i)] = W
            init_param['b'+str(i)] = b
            total_param += num_nodes[i]*(num_nodes[i-1]+1)
            print(f"W{i} shape: {W.shape}")
            print(f"b{i} shape: {b.shape}")
        print(f"Total Parameters: {total_param}")
        return init_param
    
    def initialise_adam_cache(self):
        adam_cache = {}
        for i in range(1, self.layers+1): 
            adam_cache['V_W'+str(i)] = 0
            adam_cache['V_b'+str(i)] = 0
            adam_cache['S_W'+str(i)] = 0
            adam_cache['S_b'+str(i)] = 0
        self.adam_cache = adam_cache


    def relu(self, Z, derivative=False, d_A=0):
        if not derivative:
            A = Z*((Z > 0).astype(int))
            return A
        else:
            d_Z = d_A*((Z>0).astype(int))
            return d_Z
        
    def lrelu(self, Z, derivative=False, d_A=0):
        if not derivative:
            A = np.where(Z<0, 0.01*Z, Z)
            return A
        else:
            d_Z = ((Z>0).astype(int))
            d_Z = d_A*np.where(d_Z==0, 0.01, d_Z)
            return d_Z
        
    def sigmoid(self, Z, derivative=False, d_A=0):
        A = 1/(1 + np.exp(-Z))
        if not derivative:
            return A
        else:
            d_Z = d_A*A*(1-A)
            return d_Z
        
    def tanh(self, Z, derivative=False, d_A=0):
        A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        if not derivative:
            return A
        else:
            d_Z = d_A*(1-A*A)
            return d_Z
        
    def softmax(self, Z, derivative=False, d_A=0):
        A = np.exp(Z)/np.sum(np.exp(Z), axis=0)
        if not derivative:
            return A
        else:
            d_Z = A*(d_A - np.sum(d_A*A, axis=0))
            return d_Z
        
    def linear(self, Z, derivative=False, d_A=0):
        if not derivative:
            return Z
        else:
            return d_A
        
    def activation_fn(self, name, Z, derivative=False, d_A=0):
        out = eval(f"self.{name}(Z, derivative, d_A)")
        return out
    
    def forward_pass(self, param, X, return_cache=False):
        if X.ndim == 1: #happens while performing froward prop for a single examples
            X = X.reshape((len(X), 1))

        cache = {'A0': X}

        for i in range(1, self.layers+1):
            cache['Z'+str(i)] = param['W'+str(i)]@cache['A'+str(i-1)] + param['b'+str(i)]
            cache['A'+str(i)] = self.activation_fn(self.activations[i-1], cache['Z'+str(i)])
        if return_cache:
            return cache
        else:
            return cache['A'+str(self.layers)]
        
    def compute_cost(self, param, experiences, gamma, print_opt=True):
        states, actions, rewards, next_states, done_vals = experiences
        Y = self.generate_target(next_states, rewards, done_vals, gamma)
        
        Y_hat = self.forward_pass(param, states)

        if actions.ndim != 1: actions = actions.reshape(actions.shape[1])
        m = len(actions)
        #Y_hat_filtered = Y_hat[np.column_stack((actions, tuple(range(Y_hat.shape[1]))))].reshape((1, Y_hat.shape[1]))
        #Y_hat_filtered = np.zeros((1, Y_hat.shape[1]))
        Y_hat_filtered = Y_hat[actions, range(Y_hat.shape[1])].reshape((1, Y_hat.shape[1]))
        cost = (1/(2*m))*np.sum((Y-Y_hat_filtered)**2, axis=1)

        if print_opt:
            print(f"Cost: {cost}")

        return cost
    
    def return_seed(self, A_L, Y, actions):
        m = Y.shape[1]
        if actions.ndim != 1: actions = actions.reshape(actions.shape[1])

        #mask = np.column_stack((actions, tuple(range(Y.shape[1]))))
        rows = actions
        columns = np.array(range(Y.shape[1]))
        seed = np.zeros_like(A_L)
        seed[rows, columns] = (1/m)*(A_L[rows, columns].reshape((1, Y.shape[1])) - Y)
        return seed
    
    def generate_target(self, next_states, rewards, done_vals, gamma):
        max_qsa = np.max(self.forward_pass(self.init_target_param, next_states), axis=0, keepdims=True)

        target_y = rewards.reshape(max_qsa.shape) + gamma*max_qsa*(1-done_vals.reshape(max_qsa.shape))
        return target_y
    
    def backward_pass(self, param, X, Y, actions):
        if X.ndim == 1: X = X.reshape((len(X), 1))
        if Y.ndim == 1: Y = Y.reshape((len(Y), 1))

        forward_cache = self.forward_pass(param, X, return_cache=True)
        derivatives = {}
        m = X.shape[1]
        A_L = forward_cache['A'+str(self.layers)]
        d_A = self.return_seed(A_L, Y, actions)
        for i in range(self.layers, 0, -1):
            Z = forward_cache['Z'+str(i)]
            A_prev = forward_cache['A'+str(i-1)]
            W = param['W'+str(i)]

            d_Z = self.activation_fn(self.activations[i-1], Z, derivative=True, d_A=d_A)
            
            d_W = d_Z@(A_prev.T)
            d_b = np.sum(d_Z, axis=1, keepdims=True)
            d_A = (W.T)@d_Z

            derivatives['d_W'+str(i)] = d_W 
            derivatives['d_b'+str(i)] = d_b
        return derivatives
    
    def update_step(self, experiences, gamma, alpha, beta1=0.9, beta2=0.999, E=1e-08):
        states, actions, rewards, next_states, done_vals = experiences
        target_y = self.generate_target(next_states, rewards, done_vals, gamma)

        try: self.step_num += 1
        except: self.step_num = 1

        derivatives = self.backward_pass(self.train_original_param, states, target_y, actions)

        for i in range(self.layers, 0, -1):
            d_W = derivatives['d_W'+str(i)]
            d_b = derivatives['d_b'+str(i)]
            W = self.train_original_param['W'+str(i)]
            b = self.train_original_param['b'+str(i)]

            V_W = self.adam_cache['V_W'+str(i)]
            V_b = self.adam_cache['V_b'+str(i)]
            S_W = self.adam_cache['S_W'+str(i)]
            S_b = self.adam_cache['S_b'+str(i)]

            V_W = beta1*V_W + (1-beta1)*d_W
            V_b = beta1*V_b + (1-beta1)*d_b
            S_W = beta2*S_W + (1-beta2)*(d_W**2)
            S_b = beta2*S_b + (1-beta2)*(d_b**2)
            V_W_c, V_b_c = V_W/(1-beta1**self.step_num), V_b/(1-beta1**self.step_num)
            S_W_c, S_b_c = S_W/(1-beta2**self.step_num), S_b/(1-beta2**self.step_num)

            #update the parameters of ith layer

            W = W - ((alpha/(np.sqrt(S_W_c)+E))*V_W_c)
            b = b - ((alpha/(np.sqrt(S_b_c)+E))*V_b_c)

            #writing values back to the respective caches
            self.train_original_param['W'+str(i)] = W
            self.train_original_param['b'+str(i)] = b

            self.adam_cache['V_W'+str(i)] = V_W
            self.adam_cache['V_b'+str(i)] = V_b
            self.adam_cache['S_W'+str(i)] = S_W
            self.adam_cache['S_b'+str(i)] = S_b

    def update_target_network(self, tau):
        for i in range(1, self.layers+1):
            self.train_target_param['W'+str(i)] = tau*self.train_original_param['W'+str(i)] + (1.0-tau)*self.train_target_param['W'+str(i)]
            self.train_target_param['b'+str(i)] = tau*self.train_original_param['b'+str(i)] + (1.0-tau)*self.train_target_param['b'+str(i)]

    


    
