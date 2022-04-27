#-*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import math
import pandas as pd
import networkx as nx
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict
# Compute a sequence's reward

def reward(data,coords, tour, x_dim, batch_size):
    """Reward function. Compute the total distance for a tour, given the
    coordinates of each city and the tour indexes.

    Args:
        coords (torch.Tensor): Tensor of size [batch_size, seq_len, dim],
            representing each city's coordinates.
        tour (torch.Tensor): Tensor of size [batch_size, seq_len + 1],
            representing the tour's indexes (comes back to the first city).

    Returns:
        float: Reward for this tour.
    """
    reward1= []
    #app_input=pd.DataFrame(inputs)
    app_input=coords.cpu()
    app_input =app_input.numpy()

    for i in range (batch_size):
       #cost1 = cost (tour[i])
       #reward1.append(cost1)
        in_put=tour
        #print(in_put)
        #e_w=tf.zeros([1,],tf.int64)
        e_w=0
        #reward1 = tf.zeros([128],tf.int64)
        #print(self.app)
        gg=nx.from_numpy_array(app_input)
        app = nx.to_pandas_edgelist(gg)
        #print(app)
        #aaa=pd.DataFrame(app_input[i])
        #app=nx.to_pandas_edgelist(nx.from_pandas_adjacency(aaa))
        app_len= len(app)
        for n in range (app_len):

            a=np.argwhere(in_put==app.source[n])
            b=np.argwhere(in_put==app.target[n])

            x1=a//x_dim
            y1=a%x_dim

            x2=b//x_dim
            y2=b%x_dim
            #print(a,b)
            c=abs(x1-x2)+abs(y1-y2)

            e=c*app.weight[n]

            e_w=e_w+e[0,]
        #elf.cost1=tf.reduce_sum(e_w)
        reward1.append(e_w)
        #print(reward1)
    #reward1= torch.reshape(torch.tensor(reward1, dtype=torch.float), (-1,))
    #reward = tf.cast(reward1,tf.float32)
    
    return reward1[0] # reward

# Swap city[i] with city[j] in sequence
def swap2opt(tsp_sequence,i,j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0) # flip or swap ?
    return new_tsp_sequence

# One step of 2opt = one double loop and return first improved sequence
def step2opt(input_app, tsp_sequence,x_dim):
    seq_length = tsp_sequence.shape[0]
    distance = reward(input_app,tsp_sequence, x_dim, 1)
    for i in range(1,seq_length-1):
        for j in range(i+1,seq_length):
            new_tsp_sequence = swap2opt(tsp_sequence,i,j)
            new_distance = reward(input_app,new_tsp_sequence, x_dim,1)
            if new_distance < distance:
                return new_tsp_sequence, new_distance
    return tsp_sequence, distance


class DataGenerator(object):

    def __init__(self):
        pass

    def next_step(self,graph, previous, current, p, q):
        neighbors = list(graph.neighbors(current))

        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(graph[current][neighbor]["weight"] / p)
            elif graph.has_edge(neighbor, previous):
                # The probability of visiting a local node.
                weights.append(graph[current][neighbor]["weight"])
            else:
                # Control the probability to move forward.
                weights.append(graph[current][neighbor]["weight"] / q)

        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def random_walk(self,graph, num_walks, num_steps, p, q):
        walks = []
        nodes = list(graph.nodes())
        # Perform multiple iterations of the random walk.
        for walk_iteration in range(num_walks):
            random.shuffle(nodes)

            for node in nodes:
                # Start the walk with a random node from the graph.
                walk = [node]
                # Randomly walk for num_steps.
                while len(walk) < num_steps:
                    current = walk[-1]
                    previous = walk[-2] if len(walk) > 1 else None
                    # Compute the next node to visit.
                    next = self.next_step(graph, previous, current, p, q)
                    walk.append(next)
                # Replace node ids (movie ids) in the walk with token ids.
                # Add the walk to the generated sequence.
                walks.append(walk)

        return walks

    # def gen_instance(self, max_length, dimension, seed=0): # Generate random TSP instance
    #     if seed!=0: np.random.seed(seed)
    #     sequence = np.random.rand(max_length, dimension) # (max_length) cities with (dimension) coordinates in [0,1]
    #     pca = PCA(n_components=dimension) # center & rotate coordinates
    #     sequence = pca.fit_transform(sequence) 
    #     return sequence

    def generate_examples(self,sequences, window_size, num_negative_samples, vocabulary_size):
        example_weights = defaultdict(int)
        # Iterate over all sequences (walks).
        for sequence in sequences:
            # Generate positive and negative skip-gram pairs for a sequence (walk).
            pairs, labels = keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocabulary_size,
                window_size=window_size,
                negative_samples=num_negative_samples,
            )
            for idx in range(len(pairs)):
                pair = pairs[idx]
                label = labels[idx]
                target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
                if target == context:
                    continue
                entry = (target, context, label)
                example_weights[entry] += 1

        targets, contexts, labels, weights = [], [], [], []
        for entry in example_weights:
            weight = example_weights[entry]
            target, context, label = entry
            targets.append(target)
            contexts.append(context)
            labels.append(label)
            weights.append(weight)

        return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)

    def create_dataset(self,targets, contexts, labels, weights, batch_size):
        inputs = {
            "target": targets,
            "context": contexts,
        }
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
        dataset = dataset.shuffle(buffer_size=batch_size * 2)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_model(self,vocabulary_size, embedding_dim):
        inputs = {
            "target": layers.Input(name="target", shape=(), dtype="int32"),
            "context": layers.Input(name="context", shape=(), dtype="int32"),
        }
        # Initialize item embeddings.
        embed_item = layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="item_embeddings",
        )
        # Lookup embeddings for target.
        target_embeddings = embed_item(inputs["target"])
        # Lookup embeddings for context.
        context_embeddings = embed_item(inputs["context"])
        # Compute dot similarity between target and context embeddings.
        logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
            [target_embeddings, context_embeddings]
        )
        # Create the model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def getEmbeddings(self,num_walks,walk_max_length,G,embedding_dim,num_epochs,prob_p,prob_q,num_negative_samples):
        walks=self.random_walk(G,num_walks,walk_max_length,prob_p,prob_q)
        targets, contexts, labels, weights = self.generate_examples(
                sequences=walks,
                window_size=3,
                num_negative_samples=num_negative_samples,
                vocabulary_size=len(list(G.nodes)),
            )
        dataset = self.create_dataset(
                targets=targets,
                contexts=contexts,
                labels=labels,
                weights=weights,
                batch_size=8,
            )
        learning_rate=0.005
        model = self.create_model(len(list(G.nodes)), embedding_dim)
        model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )
        model.fit(dataset, epochs=num_epochs)
        A = model.get_layer("item_embeddings").get_weights()[0]
        A=np.array(A)
        return A
    def train_batch(self, batch_size,prob_p,prob_q,num_negative_samples, max_length,num_walks,walk_max_length, dimension, x_dim, y_dim,embedding_dim,num_epochs): # Generate random batch for training procedure
        input_batch = []
        input_batch_org=[]
        Traffic=['./mpeg.csv', './mwd.csv', './263enc.csv']
        '''
        app=[]
        a=np.loadtxt("foo.txt", delimiter=",")
        for i in range (batch_size):
            d=a[i].astype('float32')
            d=d.reshape(12,12)
            #gg=nx.from_numpy_array(d)
            input_batch.append(d)
            #app.append(nx.to_pandas_edgelist(gg))       
           
        '''
        size1 = x_dim*y_dim #mesh size 4x3
        for _ in range(batch_size):        
            size = random.randint((size1 - 4),size1)

            G=nx.connected_watts_strogatz_graph(size, 5, 0.5, tries=100, seed=None)
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = random.randint(1,1000)
            #B=nx.to_numpy_array(G)
            app= nx.to_pandas_edgelist(G)
            A=np.zeros((size1,size1),dtype='float')
            for i in range (len(app)):
                A[app.source[i]][app.target[i]]= app.weight[i]
            input_batch_org.append(A)
            A=self.getEmbeddings(num_walks,walk_max_length,G,embedding_dim,num_epochs,prob_p,prob_q,num_negative_samples)
            if A.shape[0]!=size1:
                temp=np.zeros((size1-A.shape[0],embedding_dim),dtype='float')
                A=np.concatenate((A,temp),axis=0)
            input_batch.append(A)
        # A=np.zeros((size1,size1),dtype='float')
        # traffic1 = random.randint(0,2)
        # app = pd.read_csv(Traffic[traffic1])
        # for i in range ( len(app)):
        #     A[app.source[i]][app.target[i]]= app.weight[i]
        # input_batch_org.append(A)
        # G=nx.from_numpy_array(A)
        # A=self.getEmbeddings(num_walks,walk_max_length,G,embedding_dim,num_epochs)
        # input_batch.append(A)
        return input_batch,input_batch_org

    def test_batch(self, batch_size, max_length, dimension, x_dim, y_dim, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        #input_ = self.gen_instance(max_length, dimension, seed=seed) # Generate random TSP instance
        mp1=np.random.permutation(max_length)
        x_index=mp1//x_dim
        y_index=mp1%x_dim
        input_=np.column_stack((x_index, y_index))
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

    def loop2opt(self,  input_app, tsp_sequence, x_dim, max_iter=10000): # Iterate step2opt max_iter times (2-opt local search)
        
        best_reward = reward(input_app,tsp_sequence, x_dim,1)
        new_tsp_sequence = np.copy(tsp_sequence)
        for _ in range(max_iter): 
            new_tsp_sequence, new_reward = step2opt(input_app, new_tsp_sequence,x_dim)
            if new_reward < best_reward:
                best_reward = new_reward
            else:
                break
        return new_tsp_sequence, best_reward
'''
    def visualize_2D_trip(self, trip): # Plot tour
        plt.figure(1)
        colors = ['red'] # First city red
        for i in range(len(trip)-1):
            colors.append('blue')
            
        plt.scatter(trip[:,0], trip[:,1],  color=colors) # Plot cities
        tour=np.array(list(range(len(trip))) + [0]) # Plot tour
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--")

        plt.xlim(-0.75,5)
        plt.ylim(-0.75,5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def visualize_sampling(self, permutations): # Heatmap of permutations (x=cities; y=steps)
        max_length = len(permutations[0])
        grid = np.zeros([max_length,max_length]) # initialize heatmap grid to 0

        transposed_permutations = np.transpose(permutations)
        for t, cities_t in enumerate(transposed_permutations): # step t, cities chosen at step t
            city_indices, counts = np.unique(cities_t,return_counts=True,axis=0)
            for u,v in zip(city_indices, counts):
                grid[t][u]+=v # update grid with counts from the batch of permutations

        fig = plt.figure(1) # plot heatmap
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(grid, interpolation='nearest', cmap='gray')
        plt.colorbar()
        plt.title('Sampled permutations')
        plt.ylabel('Time t')
        plt.xlabel('City i')
        plt.show()
    def application(self):
        a=np.loadtxt("train.txt", delimiter=",")
        app=[]
        for i in range (len(a)):
            d=a[i].astype('int64')
            #d=tf.convert_to_tensor(d.reshape(12,12))
            d=d.reshape(12,12)
            gg=nx.from_numpy_array(d)
            app.append(nx.to_pandas_edgelist(gg))
        return app
'''
class SimulatedAnnealing:
    def __init__(self, coords, curr_solution, x_dim, temp, alpha, stopping_temp, stopping_iter):
        ''' animate the solution over time

            Parameters
            ----------
            coords: array_like
                list of coordinates
            temp: float
                initial temperature
            alpha: float
                rate at which temp decreases
            stopping_temp: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates

        '''
        self.x_dim = x_dim
        self.coords = coords
        self.sample_size = len(curr_solution)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.curr_solution = curr_solution
        self.best_solution = curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = reward(self.coords, self.best_solution, self.x_dim,1)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        #print('Intial weight: ', self.curr_weight)



    def acceptance_probability(self, candidate_weight):
        '''
        Acceptance probability as described in:
        https://stackoverflow.com/questions/19757551/basics-of-simulated-annealing-in-python
        '''
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        '''
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        '''
        candidate_weight = reward(self.coords,candidate, self.x_dim,1)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        '''
        Annealing process with 2-opt
        described here: https://en.wikipedia.org/wiki/2-opt
        '''
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)

            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

        #print('Minimum weight: ', self.min_weight,'Best Solution: ', self.best_solution)
        #print('Improvement: ', round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')
        return self.min_weight, self.best_solution
