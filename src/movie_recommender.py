import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import urllib.request
import zipfile

from scipy.linalg import cho_factor, cho_solve

class MovieRecommender:

    ###########################################################################
    # Initialization of data structures and index lists
    ###########################################################################

    def __init__(self, small_dataset: bool):
        # Useful properties
        self.k = 0
        self.trained = False
        self.small_dataset = small_dataset

        # Sort data w.r.t. users (u) and items (v)
        self.download_data()
        self.init_ratings_and_movies_data()

        # Initialize index lists and data structures
        self.init_index_lists()
        self.U_R_sorted_by_v = np.array(list(zip(self.users_v, self.ratings_v))).astype(int)
        self.V_R_sorted_by_u = np.array(list(zip(self.items_u, self.ratings_u))).astype(int)
        print("\rMovie Recommender initialized.\t\t\t", end="\n")

    def download_data(self):
        url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
        filename = 'ml-25m.zip'
        dirname = 'ml-25m'

        if (self.small_dataset):
            url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
            filename = 'ml-latest-small.zip'
            dirname = 'ml-latest-small'
        
        # Check if the dataset is already downloaded and extracted
        if not os.path.isdir(dirname):
            print(f"\rDownloading {dirname} ...", end="")
            urllib.request.urlretrieve(url, filename)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(filename)
            print(f"\rFinished downloading data.\t\t", end="")
    
    def init_ratings_and_movies_data(self):
        ratings_dir = 'ml-25m/ratings.csv'
        movies_dir = 'ml-25m/movies.csv'
        if (self.small_dataset):
            ratings_dir = 'ml-latest-small/ratings.csv'
            movies_dir = 'ml-latest-small/movies.csv'
        
        users, items, ratings = np.loadtxt(
                ratings_dir, 
                delimiter=',', 
                usecols=(0, 1, 2),
                skiprows=1,
                unpack=True
        )
        self.num_ratings = ratings.shape[0]
        
        users = np.unique(users.astype(int), return_inverse=True)[1] 
        uniq_v, items = np.unique(items.astype(int), return_inverse=True)
        ratings = ratings.astype(float)

        idx_u = np.argsort(users)   # Indices of sorted users
        idx_v = np.argsort(items)   # Indices of sorted items

        self.users_u = users[idx_u]      # Users sorted w.r.t. users
        self.users_v = users[idx_v]      # Users sorted w.r.t. items
        self.items_u = items[idx_u]      # Items sorted w.r.t. users
        self.items_v = items[idx_v]      # Items sorted w.r.t. items
        self.ratings_u = ratings[idx_u]  # Ratings sorted w.r.t. users
        self.ratings_v = ratings[idx_v]  # Ratings sorted w.r.t. items

        movies_data = pd.read_csv(movies_dir)
        movies_data = movies_data[movies_data['movieId'].isin(uniq_v)]
        movies_data = movies_data.sort_values("movieId")
        movies_data["movieId"] = np.unique(movies_data["movieId"], return_inverse=True)[1]
        self.movie_names  = movies_data["title"]
        self.movie_genres = movies_data["genres"]
        # self.movie_df = movies_data

    def init_index_lists(self):
        # Index lists
        self.U_start = [0]; self.U_end = []
        self.V_start = [0]; self.V_end = []

        # Users index
        prev_user = self.users_u[0]
        for m, user in enumerate(self.users_u):
            if (user != prev_user):
                self.U_start.append(m)
                self.U_end.append(m)
                prev_user = user
        self.U_end.append(self.users_u.shape[0])

        # Items index
        prev_item = self.items_v[0]
        for n, item in enumerate(self.items_v):
            if (item != prev_item):
                self.V_start.append(n)
                self.V_end.append(n)
                prev_item = item
        self.V_end.append(self.items_v.shape[0])

    ###########################################################################
    # Training a new movie recommender model
    ###########################################################################

    def train_model_als(self, max_epoch: int = 10, lambda_: float = 0.1,
                        tau: float = 0.1, k: int = 20, parallel: bool = True):
        # Init hyper parameters and end of range values
        M = len(self.U_start)
        N = len(self.V_start)
        self.ll = lambda_
        self.tau = tau
        self.k = k

        # Trait vectors and bias
        dev = 2.5/np.sqrt(k)
        self.U   = np.random.normal(loc=0.0, scale=dev, size=M*k).reshape(M, k)
        self.V   = np.random.normal(loc=0.0, scale=dev, size=N*k).reshape(N, k)
        self.b_u = np.zeros(shape=(M,))
        self.b_v = np.zeros(shape=(N,))
        
        # Loss/error lists
        self.loss_values = []
        self.rmse_values = []

        # Split ranges for multiprocessing
        M_chunks = range(M)
        N_chunks = range(N)
        if (parallel):
            cores = mp.cpu_count()
            M_chunks = self.split_range(range(M), self.U_end, cores)
            N_chunks = self.split_range(range(N), self.V_end, cores)

        # Main loop
        epoch = 0
        total_time_start = time.perf_counter()
        print("Alternating Least Squares:")

        while (epoch != max_epoch):
            time_start = time.perf_counter()

            if (parallel):
                print("\r0/4", end="")
                with mp.Pool(cores) as p:
                    results = p.map(self.update_user_bias, M_chunks)
                    for result in results:
                        self.b_u[result[0]] = result[1]

                print("\r1/4", end="")
                with mp.Pool(cores) as p:
                    results = p.map(self.update_user_trait_vec, M_chunks)
                    for result in results:
                        self.U[result[0]] = result[1]

                print("\r2/4", end="")
                with mp.Pool(cores) as p:
                    results = p.map(self.update_item_bias, N_chunks)
                    for result in results:
                        self.b_v[result[0]] = result[1]

                print("\r3/4", end="")
                with mp.Pool(cores) as p:
                    results = p.map(self.update_item_trait_vec, N_chunks)
                    for result in results:
                        self.V[result[0]] = result[1]

                print("\r4/4", end="")
                with mp.Pool(cores) as p:
                    results = p.map(self.compute_loss, M_chunks)
                    squared_error = 0.0
                    for result in results:
                        squared_error += result
                    loss = -(self.ll/2) * squared_error
                    rmse = np.sqrt(squared_error/(self.num_ratings))
                    self.loss_values.append(loss)
                    self.rmse_values.append(rmse)
            else:
                print("\r0/4", end="")
                _, val = self.update_user_bias(M_chunks)
                self.b_u[M_chunks] = val

                print("\r1/4", end="")
                _, val = self.update_user_trait_vec(M_chunks)
                self.U[M_chunks] = val
                
                print("\r2/4", end="")
                _, val = self.update_item_bias(N_chunks)
                self.b_v[N_chunks] = val
                
                print("\r3/4", end="")
                _, val = self.update_item_trait_vec(N_chunks)
                self.V[N_chunks] = val

                print("\r4/4", end=""); squared_error = self.compute_loss(M_chunks)
                loss = -(self.ll/2) * squared_error
                rmse = np.sqrt(squared_error/(self.num_ratings))
                self.loss_values.append(loss)
                self.rmse_values.append(rmse)
            
            epoch += 1
            time_end = time.perf_counter()
            out_str = f"\r  Epoch {epoch}: "
            out_str += f"loss = {round(self.loss_values[-1], 4)}, "
            out_str += f"rmse = {round(self.rmse_values[-1], 4)}, "
            out_str += f"time elapsed = {np.round(time_end - time_start, 4)}s"
            print(out_str, end="\n")

        self.trained = True
        self.save_model(max_epoch)
        total_time = time.perf_counter() - total_time_start
        print(f"Finished. Time elapsed = {total_time}s", end="\n")

    def split_range(self, range_, idx_array, cores):
        chunks = []
        range_list = list(range_)
        prev_idx, idx = 0, 0
        for i in range(1, cores):
            stopping_val = self.num_ratings*i/cores
            while (idx_array[idx] < stopping_val):
                idx += 1
            chunks.append(range_list[prev_idx:idx])
            prev_idx = idx
        chunks.append(range_list[idx:])
        return chunks

    def update_user_bias(self, range_M):
        for m in range_M:
            b_u_m = np.sum(self.V_R_sorted_by_u[self.U_start[m]:self.U_end[m], 1] \
                - self.U[None, m] @ self.V[self.V_R_sorted_by_u[self.U_start[m]:self.U_end[m], 0]].T \
                - self.b_v[None, self.V_R_sorted_by_u[self.U_start[m]:self.U_end[m], 0]])
            self.b_u[m] = b_u_m / (self.U_end[m] - self.U_start[m])
        return range_M, self.b_u[range_M]

    def update_user_trait_vec(self, range_M):
        f = self.k
        for m in range_M:
            arr, vec = np.zeros(shape=(f, f)), np.zeros(shape=(f,))
            for n in range(self.U_start[m], self.U_end[m]):
                arr += np.outer(self.V[self.V_R_sorted_by_u[n][0]], self.V[self.V_R_sorted_by_u[n][0]])
                vec += self.V[self.V_R_sorted_by_u[n][0]] * (self.V_R_sorted_by_u[n][1] - self.b_u[m] - self.b_v[self.V_R_sorted_by_u[n][0]])
            arr += self.tau*np.identity(f)
            arr = self.inv(arr)
            vec = self.ll*vec
            self.U[m] = np.dot(arr, vec)
        return range_M, self.U[range_M]

    def update_item_bias(self, range_N):
        for n in range_N:    
            b_v_n = 0.0
            for m in range(self.V_start[n], self.V_end[n]):
                b_v_n += self.U_R_sorted_by_v[m][1] - np.dot(self.U[self.U_R_sorted_by_v[m][0]], self.V[n]) - self.b_u[self.U_R_sorted_by_v[m][0]]
            self.b_v[n] = b_v_n / (self.V_end[n] - self.V_start[n])
        return range_N, self.b_v[range_N]

    def update_item_trait_vec(self, range_N):
        f = self.k
        for n in range_N:
            arr, vec = np.zeros(shape=(f, f)), np.zeros(shape=(f,))
            for m in range(self.V_start[n], self.V_end[n]):
                arr += np.outer(self.U[self.U_R_sorted_by_v[m][0]], self.U[self.U_R_sorted_by_v[m][0]])
                vec += self.U[self.U_R_sorted_by_v[m][0]] * (self.U_R_sorted_by_v[m][1] - self.b_u[self.U_R_sorted_by_v[m][0]] - self.b_v[n])
            arr += self.tau*np.identity(f)
            arr = self.inv(arr)
            vec = self.ll*vec
            self.V[n] = np.dot(arr, vec)
        return range_N, self.V[range_N]

    def compute_loss(self, range_M):
        squared_err = 0.0
        for m in range_M:
            for n in range(self.U_start[m], self.U_end[m]):
                item_idx = self.V_R_sorted_by_u[n][0]
                squared_err += (self.V_R_sorted_by_u[n][1] - np.inner(self.U[m], self.V[item_idx]) - self.b_u[m] - self.b_v[item_idx])**2
        return squared_err
    
    def inv(self, A):
        L, lower = cho_factor(A, overwrite_a=True)
        B = np.eye(A.shape[0])
        X = cho_solve((L, lower), B, overwrite_b=True)
        return X

    def save_model(self, max_epoch):
        directory = "results/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "big/"
        directory += f"ll = {self.ll}, tau = {self.tau}, k = {self.k}, iters = {max_epoch}/"

        if (not os.path.isdir(directory)):
            os.mkdir(directory)

        np.savetxt(f'{directory}U.out', self.U, delimiter=',')
        np.savetxt(f'{directory}V.out', self.V, delimiter=',')
        np.savetxt(f'{directory}u_b.out', self.b_u, delimiter=',')
        np.savetxt(f'{directory}v_b.out', self.b_v, delimiter=',')
    
    ###########################################################################
    # Loading a pre-existing movie recommender model
    ###########################################################################

    def load_model(self, max_epoch, ll, tau, k):
        directory = "results/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "big/"
        directory += f"ll = {ll}, tau = {tau}, k = {k}, iters = {max_epoch}/"

        if (not os.path.isdir(directory)):
            raise Exception('Model does not exist ...')
        
        self.ll = ll
        self.tau = tau
        self.k = k
        self.U =   np.loadtxt(f'{directory}U.out', delimiter=',')
        self.V =   np.loadtxt(f'{directory}V.out', delimiter=',')
        self.b_u = np.loadtxt(f'{directory}u_b.out', delimiter=',')
        self.b_v = np.loadtxt(f'{directory}v_b.out', delimiter=',')
        self.trained = True
        
    ###########################################################################
    # Predicting ratings for a new user
    ###########################################################################

    def predict(self, item_ids, item_ratings):
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        assert len(item_ids) == len(item_ratings)

        user_bias, user_vec = 0, np.random.normal(loc=0.0, scale=2.5/np.sqrt(self.k), size=self.k)

        for _ in range(5):
            b_v_m = 0.0
            for i, n in enumerate(item_ids):
                b_v_m += item_ratings[i] - np.inner(user_vec, self.V[n]) - self.b_v[n]
            user_bias = b_v_m / len(item_ids)

            arr, vec = np.zeros(shape=(self.k, self.k)), np.zeros(shape=(self.k,))
            for i, n in enumerate(item_ids):
                arr += np.outer(self.V[n], self.V[n])
                vec += self.V[n] * (item_ratings[i] - user_bias - self.b_v[n])
            arr += self.tau*np.identity(self.k)
            arr = self.inv(arr)
            vec = self.ll*vec
            user_vec = np.dot(arr, vec)

        predictions = np.zeros(shape=(self.V.shape[0],))
        for i in range(self.V.shape[0]):
            predictions[i] = np.inner(user_vec, self.V[i]) + user_bias + 0.05 * self.b_v[i]

        idx = np.flip(np.argsort(predictions))[:10]
        for i in idx:
            print(f"{self.movie_names[i]}: {predictions[i]}")

    ###########################################################################
    # Methods for plotting distributions of data and loss/error over time
    ###########################################################################

    def plot_user_item_distribution(self):
        num_ratings_per_user = []
        num_ratings_per_item = []

        for i in range(len(self.U_start)):
            num_ratings_per_user.append(self.U_end[i] - self.U_start[i])

        for j in range(len(self.V_start)):
            num_ratings_per_item.append(self.V_end[j] - self.V_start[j])

        U = np.zeros(max(num_ratings_per_user) + 1)
        for u in num_ratings_per_user:
            U[u] += 1

        V = np.zeros(max(num_ratings_per_item) + 1)
        for v in num_ratings_per_item:
            V[v] += 1

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(range(V.shape[0]), V, color=[0,0,0.69], s=1.9, label='items')
        plt.scatter(range(U.shape[0]), U, color=[0,0.69,0], s=1.9, label='users')
        plt.legend()
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Distribution of users and items.')
        plt.show()
            
    def plot_loss(self):
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")

        plt.figure()
        plt.plot(range(1, len(self.loss_values) + 1), np.array(self.loss_values).reshape(len(self.loss_values), ))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss function over time")

        plt.figure()
        plt.plot(range(1, len(self.rmse_values) + 1), np.array(self.rmse_values).reshape(len(self.rmse_values), ))
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE function over time")

        plt.show()
    
    def plot2D(self):
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        if (self.k != 2):
            raise Exception("Dimensionality of latent (trait) vectors != 2.")
        
        genres = []
        all_genres, genre_counts = np.unique(self.movie_genres, return_counts=True)

        for i in range(len(all_genres)):
            if (genre_counts[i] > 100):
                genres.append(all_genres[i])
        
        plt.figure()
        for genre in all_genres:
            idx = np.where(self.movie_genres == genre)[0]
            plt.scatter(self.V[idx, 0], self.V[idx, 1], cmap='cool', label=genre)
        plt.legend()
        plt.show()


def main():
    model = MovieRecommender(small_dataset=False)
    # model.train_model_als(max_epoch=10, lambda_=0.01, tau=0.1, k=20, parallel=True)
    model.load_model(max_epoch=10, ll=0.01, tau=0.1, k=20)
    print(f"Predicting {model.movie_names[23398]}")
    model.predict([23398], [5.0])

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()