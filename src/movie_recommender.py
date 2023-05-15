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
    """
    A class that represents a movie recommender system.
    """

    def __init__(self, small_dataset: bool):
        """
        Initializes data structures and index lists.

        Args:
         - small_dataset (bool): A flag indicating whether to use
                                 the small dataset or not.

        Returns: None.
        """
        self.k = 0
        self.trained = False
        self.small_dataset = small_dataset

        # Sort data w.r.t. users (u) and items (v)
        self.download_data()
        print(f"Initializing ...")

        self.init_ratings_and_movies_data()
        self.init_index_lists()

        dt = np.dtype([('users_v', int), ('ratings_v', float)])
        dt2 = np.dtype([('items_u', int), ('ratings_u', float)])

        # Create the arrays with the specified data types
        self.U_R_sorted_by_v = np.array(
            list(zip(self.users_v, self.ratings_v)),
            dtype=dt
        )
        self.V_R_sorted_by_u = np.array(
            list(zip(self.items_u, self.ratings_u)),
            dtype=dt2
        )
        print("Movie Recommender initialized.")

    def download_data(self):
        """
        Downloads the movie dataset if it hasn't been downloaded already.

        Returns: None.
        """
        url = 'http://files.grouplens.org/'
        url += 'datasets/movielens/ml-25m.zip'
        filename = 'ml-25m.zip'
        dirname = 'ml-25m'

        if self.small_dataset:
            url = 'http://files.grouplens.org/'
            url += 'datasets/movielens/ml-latest-small.zip'
            filename = 'ml-latest-small.zip'
            dirname = 'ml-latest-small'

        # Check if the dataset is already downloaded and extracted
        if not os.path.isdir(dirname):
            print(f"Downloading {dirname} ...")
            urllib.request.urlretrieve(url, filename)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(filename)
            print(f"Finished downloading data")

    def init_ratings_and_movies_data(self):
        """
        Initializes ratings and movies data.

        Returns: None.
        """
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

        idx_u = np.argsort(users)        # Indices of sorted users
        idx_v = np.argsort(items)        # Indices of sorted items

        self.users_u = np.array(users[idx_u], dtype=int)
        self.users_v = np.array(users[idx_v], dtype=int)
        self.items_u = np.array(items[idx_u], dtype=int)
        self.items_v = np.array(items[idx_v], dtype=int)
        self.ratings_u = np.array(ratings[idx_u], dtype=float)
        self.ratings_v = np.array(ratings[idx_v], dtype=float)

        movies_data = pd.read_csv(movies_dir)
        movies_data = movies_data[movies_data['movieId'].isin(uniq_v)]
        movies_data = movies_data.sort_values("movieId")
        movies_data["movieId"] = np.unique(
            movies_data["movieId"],
            return_inverse=True)[1]

        self.movie_names = np.array(movies_data["title"], dtype=str)
        self.movie_genres = np.array(movies_data["genres"], dtype=str)

        all_genre_names = []
        for g_str in self.movie_genres:
            all_genre_names.extend(g_str.split("|"))
        self.unique_genres = list(np.unique(all_genre_names))

    def init_index_lists(self):
        """
        Initialize index lists.

        Returns: None.
        """
        # Index lists
        self.U_start = [0]
        self.U_end = []
        self.V_start = [0]
        self.V_end = []

        # Users index
        prev_user = self.users_u[0]
        assert self.users_u[0] == 0
        for m, user in enumerate(self.users_u):
            if user != prev_user:
                self.U_start.append(m)
                self.U_end.append(m)
                prev_user = user

        self.U_end.append(self.users_u.shape[0])
        self.U_start = np.array(self.U_start, dtype=int)
        self.U_end = np.array(self.U_end, dtype=int)

        # Items index
        prev_item = self.items_v[0]
        assert self.items_v[0] == 0
        for n, item in enumerate(self.items_v):
            if item != prev_item:
                self.V_start.append(n)
                self.V_end.append(n)
                prev_item = item

        self.V_end.append(self.items_v.shape[0])
        self.V_start = np.array(self.V_start, dtype=int)
        self.V_end = np.array(self.V_end, dtype=int)

        # Genre index
        self.movie_genre_list = []
        for genre_str in self.movie_genres:
            genre_list, index_list = genre_str.split("|"), []
            for genre in genre_list:
                index = self.unique_genres.index(genre)
                index_list.append(index)
            self.movie_genre_list.append(index_list)

    ##########################################################################
    # Training a new movie recommender model
    ##########################################################################

    def train_model_als(
        self,
        k: int = 20,
        ll: float = 0.1,
        tau: float = 0.01,
        alpha: float = 0.01,
        beta: float = 0.01,
        max_epoch: int = 10,
        save_model: bool = True
    ):
        """
        Train the matrix factorization model using Alternating Least Squares
        (ALS) with the given hyperparameters.

        Args:
         - k (int, optional): The latent dimensionality.
                              Defaults to 20.
         - ll (float, optional): The initial variance for the model's
                                 U and V matrices.
                                 Defaults to 0.1.
         - tau (float, optional): The regularization coefficient for the
                                  user and item trait vectors.
                                  Defaults to 0.01.
         - alpha (float, optional): The regularization coefficient for the
                                    user and genre bias terms.
                                    Defaults to 0.01.
         - beta (float, optional): The regularization coefficient for the
                                   genre trait vectors.
                                   Defaults to 0.01.
         - max_epoch (int, optional): The maximum number of training epochs.
                                      Defaults to 10.
         - save_model (bool, optional): True, if the trained model is saved.
                                        Defaults to True.

        Returns: None.
        """
        # Init max vals
        M = len(self.U_start)
        N = len(self.V_start)
        num_genres = len(self.unique_genres)

        if (not self.trained):
            # Control and hyper parameters
            self.k = k
            self.ll = ll
            self.tau = tau
            self.alpha = alpha
            self.beta = beta
            self.epoch = 0

            # Trait vectors and bias terms
            std = np.sqrt(self.ll)

            self.U = np.random.normal(
                loc=np.sqrt(2.5/k), scale=std, size=M*k
            ).reshape(M, k)   # User matrix

            self.V = np.random.normal(
                loc=np.sqrt(2.5/k), scale=std, size=N*k
            ).reshape(N, k)   # Item matrix

            self.F = np.random.normal(
                loc=np.sqrt(2.5/k), scale=std, size=num_genres*k
            ).reshape(num_genres, k)   # Genre matrix

            self.b_u = np.zeros(shape=(M,))
            self.b_v = np.zeros(shape=(N,))

            # Loss/error lists
            self.loss_values = []
            self.rmse_values = []
            self.mae_values = []
        else:
            assert self.epoch > 0
            max_epoch += self.epoch

        # Split ranges for multiprocessing
        M_chunks = range(M)
        N_chunks = range(N)
        cores = mp.cpu_count()
        print(f"Running with {cores} cores")
        M_chunks = self.split_range(range(M), self.U_end, cores)
        N_chunks = self.split_range(range(N), self.V_end, cores)

        # Main loop
        total_time_start = time.perf_counter()
        print("Alternating Least Squares:")
        while (self.epoch < max_epoch):
            time_start = time.perf_counter()

            print("\r0/6", end="")
            with mp.Pool(cores) as p:
                results = p.map(self.update_user_bias, M_chunks)
                for result in results:
                    self.b_u[result[0]] = result[1]

            print("\r1/6", end="")
            with mp.Pool(cores) as p:
                results = p.map(self.update_user_trait_vec, M_chunks)
                for result in results:
                    self.U[result[0]] = result[1]

            print("\r2/6", end="")
            with mp.Pool(cores) as p:
                results = p.map(self.update_item_bias, N_chunks)
                for result in results:
                    self.b_v[result[0]] = result[1]

            print("\r3/6", end="")
            with mp.Pool(cores) as p:
                results = p.map(self.update_item_trait_vec, N_chunks)
                for result in results:
                    self.V[result[0]] = result[1]

            print("\r4/6", end="")
            self.update_genre_trait_vec(N, num_genres)

            print("\r5/6", end="")
            self.compute_loss_error(M, N, num_genres)

            print("\r6/6", end="")
            self.epoch += 1
            time_end = time.perf_counter()
            out_str = f"\r  Epoch {self.epoch}: "
            out_str += f"loss = {round(self.loss_values[-1], 2)}, "
            out_str += f"rmse = {round(self.rmse_values[-1], 4)}, "
            out_str += f"mae = {round(self.mae_values[-1], 4)}, "
            out_str += f"time elapsed = {np.round(time_end - time_start, 3)}s"
            print(out_str, end="\n")

        total_time = time.perf_counter() - total_time_start
        print(f"Finished. Time elapsed = {total_time}s", end="\n")

        self.trained = True
        if (save_model):
            self.save_model(max_epoch)

    def split_range(self, range_, idx_array, cores):
        """Divide a range into chunks and return them.

        Args:
         - range_ (range): The range that needs to be divided into chunks.
         - idx_array (np.array): The array of indices.
         - cores (int): The number of cores to be used
                        for splitting the range.

        Returns: List of np.arrays: The divided chunks of the range.
        """
        chunks = []
        range_list = list(range_)
        prev_idx, idx = 0, 0
        for i in range(1, cores):
            stopping_val = self.num_ratings*i/cores
            while (idx_array[idx] < stopping_val):
                idx += 1
            chunks.append(range_list[prev_idx:idx])
            prev_idx = idx
        chunks.append(np.array(range_list[idx:], dtype=int))
        return chunks

    def update_user_bias(self, range_M):
        """Update the user bias terms for the given range.

        Args:
         - range_M (range): The range of users for which
                            the bias needs to be updated.

        Returns: Tuple of range and np.array: The range of users and their
                                              corresponding biases.
        """
        for m in range_M:
            b_u_m = 0.0
            for n in range(self.U_start[m], self.U_end[m]):
                b_u_m += self.V_R_sorted_by_u[n][1]
                b_u_m += - np.inner(self.V[self.V_R_sorted_by_u[n][0]],
                                    self.U[m])
                b_u_m += - self.b_v[self.V_R_sorted_by_u[n][0]]
            self.b_u[m] = ((self.ll)/(
                self.alpha + self.ll*(self.U_end[m] - self.U_start[m])))*b_u_m
        return range_M, self.b_u[range_M]

    def update_user_trait_vec(self, range_M):
        """Update the trait vectors of the users for the given range.

        Args:
         - range_M (range): The range of users for which the
                            trait vectors need to be updated.

        Returns: Tuple of range and np.array: The range of users and their
                                              corresponding trait vectors.
        """
        f = self.k
        for m in range_M:
            arr, vec = np.zeros(shape=(f, f)), np.zeros(shape=(f,))
            for n in range(self.U_start[m], self.U_end[m]):
                arr += np.outer(self.V[self.V_R_sorted_by_u[n][0]],
                                self.V[self.V_R_sorted_by_u[n][0]])
                vec += self.V[self.V_R_sorted_by_u[n][0]] * (
                    self.V_R_sorted_by_u[n][1]
                    - self.b_u[m]
                    - self.b_v[self.V_R_sorted_by_u[n][0]])
            arr = arr*self.ll + self.tau*np.identity(f)
            arr = self.inv(arr)
            vec = self.ll*vec
            self.U[m] = np.dot(arr, vec)
        return range_M, self.U[range_M]

    def update_item_bias(self, range_N):
        """Update the item bias terms for the given range.

        Args:
         - range_N (range): The range of items for which the
                            bias needs to be updated.

        Returns: Tuple of range and np.array: The range of items and their
                                              corresponding biases.
        """
        for n in range_N:
            b_v_n = 0.0
            for m in range(self.V_start[n], self.V_end[n]):
                b_v_n += self.U_R_sorted_by_v[m][1]
                - np.inner(self.U[self.U_R_sorted_by_v[m][0]], self.V[n])
                - self.b_u[self.U_R_sorted_by_v[m][0]]
            self.b_v[n] = ((self.ll)/(self.alpha + self.ll*(
                self.V_end[n] - self.V_start[n])))*b_v_n
        return range_N, self.b_v[range_N]

    def update_item_trait_vec(self, range_N):
        """Update the trait vectors of the items for the given range.

        Args:
         - range_N (range): The range of items for which the trait vectors
                            need to be updated.

        Returns: Tuple of range and np.array: The range of items and their
                                              corresponding trait vectors.
        """
        f = self.k
        for n in range_N:
            arr = np.zeros(shape=(f, f))
            vec1 = np.zeros(shape=(f,))
            vec2 = np.zeros(shape=(f,))
            for m in range(self.V_start[n], self.V_end[n]):
                arr += np.outer(self.U[self.U_R_sorted_by_v[m][0]],
                                self.U[self.U_R_sorted_by_v[m][0]])
                vec1 += self.U[self.U_R_sorted_by_v[m][0]] * (
                    self.U_R_sorted_by_v[m][1]
                    - self.b_u[self.U_R_sorted_by_v[m][0]]
                    - self.b_v[n])
            for i in self.movie_genre_list[n]:
                vec2 += self.F[i]
            arr = arr*self.ll + self.tau*np.identity(f)
            arr = self.inv(arr)
            vec = self.ll*vec1 + (
                self.tau/np.sqrt(len(self.movie_genre_list[n]))
            )*vec2
            self.V[n] = np.dot(arr, vec)
        return range_N, self.V[range_N]

    def update_genre_trait_vec(self, N, num_genres):
        """Update the trait vectors of the genres.

        Args:
         - N (int): The number of movies.
         - num_genres (int): The number of genres.

        Returns: None.
        """
        for i in range(num_genres):
            # Get sum
            v_sum = np.zeros(shape=(self.k,))
            f_sum = np.zeros(shape=(self.k,))
            coeff = 0.0
            for n in range(N):
                if (i in self.movie_genre_list[n]):
                    in_sum = np.zeros(shape=(self.k,))
                    v_sum += self.V[n]
                    coeff = 1/(np.sqrt(len(self.movie_genre_list[n])))
                    for j in self.movie_genre_list[n]:
                        if (j == i):
                            continue
                        in_sum += self.F[j]
                    f_sum += coeff*in_sum
            self.F[i] = v_sum - f_sum

            # Get coeff
            coeff_sum = 0.0
            for n in range(N):
                if (i in self.movie_genre_list[n]):
                    coeff_sum += 1/(np.sqrt(len(self.movie_genre_list[n])))
            ugly_coeff = (self.tau*coeff_sum)/(
                self.tau*(coeff_sum**2) + self.beta)

            # Multiply sum with coeff
            self.F[i] = ugly_coeff * self.F[i]

    def compute_loss_error(self, M, N, num_genres):
        """Compute the loss and error of the model.

        Args:
         - M (int): The number of users.
         - N (int): The number of items.
         - num_genres (int): The number of genres.

        Returns: None.
        """
        squared_err = 0.0
        abs_err = 0.0
        reg_u_m, reg_v_n = 0.0, 0.0
        reg_b_u, reg_b_v = 0.0, 0.0
        reg_f_i = 0.0

        for m in range(M):
            for n in range(self.U_start[m], self.U_end[m]):
                item_idx = self.V_R_sorted_by_u[n][0]
                squared_err += (
                    self.V_R_sorted_by_u[n][1]
                    - np.inner(self.U[m], self.V[item_idx])
                    - self.b_u[m]
                    - self.b_v[item_idx]
                )**2
                abs_err += np.abs(
                    self.V_R_sorted_by_u[n][1]
                    - np.inner(self.U[m], self.V[item_idx])
                    - self.b_u[m]
                    - self.b_v[item_idx]
                )

        for m in range(M):
            reg_u_m += np.inner(self.U[m], self.U[m])
        reg_u_m = -(self.tau/2)*reg_u_m

        for n in range(N):
            f_sum = np.zeros(shape=(self.k,))
            for i in self.movie_genre_list[n]:
                f_sum += self.F[i]
            f_sum = (1/(np.sqrt(len(self.movie_genre_list[n])))) * f_sum
            reg_v_n += np.inner((self.V[n] - f_sum), (self.V[n] - f_sum))
        reg_v_n = -(self.tau/2)*reg_v_n

        for m in range(M):
            reg_b_u += (self.b_u[m])**2
        reg_b_u = -(self.alpha/2)*reg_b_u

        for n in range(N):
            reg_b_v += (self.b_v[n])**2
        reg_b_v = -(self.alpha/2)*reg_b_v

        for i in range(num_genres):
            reg_f_i += np.inner(self.F[i], self.F[i])
        reg_f_i = -(self.beta/2)*reg_f_i

        loss = -(self.ll/2)*squared_err + (
            reg_u_m + reg_v_n + reg_b_u + reg_b_v + reg_f_i
        )
        rmse = np.sqrt((1/self.num_ratings)*squared_err)
        mae = (1/self.num_ratings)*abs_err

        self.loss_values.append(loss)
        self.rmse_values.append(rmse)
        self.mae_values.append(mae)

    ##########################################################################
    # Save and load a movie recommender model
    ##########################################################################

    def save_model(self, max_epoch):
        """
        Saves the trained model's matrices and other data into a set of
        text files in a newly-created directory named according to the
        input hyperparameters.

        Args:
         - max_epoch (int): The maximum number of training epochs the
                            model underwent.

        Returns: None
        """
        directory = "results/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "ml-25m/"
        directory += f"k = {self.k}, ll = {self.ll}, tau = {self.tau},"
        directory += f" alpha = {self.alpha}, beta = {self.beta},"
        directory += f" iters = {max_epoch}/"

        if (not os.path.isdir(directory)):
            os.makedirs(directory)

        np.savetxt(f'{directory}U.out', self.U, delimiter=',')
        np.savetxt(f'{directory}V.out', self.V, delimiter=',')
        np.savetxt(f'{directory}u_b.out', self.b_u, delimiter=',')
        np.savetxt(f'{directory}v_b.out', self.b_v, delimiter=',')
        np.savetxt(f'{directory}F.out', self.F, delimiter=',')
        np.savetxt(f'{directory}loss_vals.out',
                   np.array(self.loss_values), delimiter=',')
        np.savetxt(f'{directory}rmse_vals.out',
                   np.array(self.rmse_values), delimiter=',')
        np.savetxt(f'{directory}mae_vals.out',
                   np.array(self.mae_values), delimiter=',')

    def load_model(self, k, ll, tau, alpha, beta, max_epoch):
        """
        Loads a previously-trained model's data from a set of text files
        saved in a directory named according to the input hyperparameters.
        Overwrites the current instance's U, V, b_u, b_v, and F matrices
        with the loaded data.

        Args:
         - k (int): The latent dimensionality.
         - ll (float): The initial variance for the model's
                       U and V matrices.
         - tau (float): The regularization coefficient for the user and
                        item trait vectors.
         - alpha (float): The regularization coefficient for the user
                          and item biases.
         - beta (float): The regularization coefficient for the genre
                         trait vectors.
         - max_epoch (int): The maximum number of training epochs the
                            model underwent.

        Returns: None
        """
        directory = "results/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "ml-25m/"
        directory += f"k = {k}, ll = {ll}, tau = {tau},"
        directory += f" alpha = {alpha}, beta = {beta},"
        directory += f" iters = {max_epoch}/"

        if (not os.path.isdir(directory)):
            raise Exception('Model does not exist ...')

        self.k = k
        self.ll = ll
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.epoch = max_epoch

        self.U = np.loadtxt(f'{directory}U.out', delimiter=',')
        self.V = np.loadtxt(f'{directory}V.out', delimiter=',')
        self.F = np.loadtxt(f'{directory}F.out', delimiter=',')
        self.b_u = np.loadtxt(f'{directory}u_b.out', delimiter=',')
        self.b_v = np.loadtxt(f'{directory}v_b.out', delimiter=',')
        self.loss_values = np.loadtxt(
            f'{directory}loss_vals.out',
            delimiter=','
        ).tolist()
        self.rmse_values = np.loadtxt(
            f'{directory}rmse_vals.out',
            delimiter=','
        ).tolist()
        self.mae_values = np.loadtxt(
            f'{directory}mae_vals.out',
            delimiter=','
        ).tolist()

        self.trained = True

    ###########################################################################
    # Predicting ratings for a new user
    ###########################################################################

    def predict(self, item_ids, item_ratings, num_predictions):
        """
        Predicts a set of user ratings for all items in the instance's V
        matrix by modeling the user as a vector in the model's latent space
        and computing the dot product between this vector and each item's
        vector in V. Prints the input item IDs, the number of items, and the
        top predicted ratings to the console.

        Args:
         - item_ids (list of int): The IDs of the items being rated by
                                   the user
         - item_ratings (list of float): The ratings given by the user
                                         to each item in item_ids
         - num_predictions (int): the number of top-rated items to display

        Returns: None
        """
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        assert len(item_ids) == len(item_ratings)

        print(f"{'-'*60}\nPredictions for:")
        for id, rating in zip(item_ids, item_ratings):
            print(f"  - {self.movie_names[id]}: {rating}"
                  + f"[{self.movie_genres[id]}]")
        print(f"{'-'*60}", end="\n")

        user_bias, user_vec = 0, np.random.normal(
            loc=np.sqrt(2.5/self.k), scale=np.sqrt(self.ll), size=self.k
        )
        for _ in range(5):
            b_v_m = 0.0
            for i, n in enumerate(item_ids):
                b_v_m += item_ratings[i] - np.inner(user_vec, self.V[n])
                - self.b_v[n]
            user_bias = b_v_m / len(item_ids)

            arr = np.zeros(shape=(self.k, self.k))
            vec = np.zeros(shape=(self.k,))
            for i, n in enumerate(item_ids):
                arr += np.outer(self.V[n], self.V[n])
                vec += self.V[n] * (
                    item_ratings[i] - user_bias - self.b_v[n]
                )
            arr = arr*self.ll + self.tau*np.identity(self.k)
            arr = self.inv(arr)
            vec = self.ll*vec
            user_vec = np.dot(arr, vec)

        assert self.V.shape[0] == self.movie_names.shape[0]

        predictions = np.zeros(shape=(self.V.shape[0],))
        for i in range(self.V.shape[0]):
            predictions[i] = np.inner(user_vec,
                                      self.V[i]) + user_bias + 0.1*self.b_v[i]

        print(f"\nTop {num_predictions} ratings:")
        idx = np.flip(np.argsort(predictions))
        for i in idx[:num_predictions]:
            print(f" - {self.movie_names[i]}: {predictions[i]}"
                  + f"[{self.movie_genres[i]}] [id = {i}]")

    ##########################################################################
    # Helper methods
    ##########################################################################

    # Cholesky inverse
    def inv(self, A):
        """
        Returns the inverse of matrix A using Cholesky decomposition.

        Args:
         - A (numpy.ndarray): The input matrix.

        Returns: numpy.ndarray: The inverse of matrix A.
        """
        L, lower = cho_factor(A, overwrite_a=True)
        B = np.eye(A.shape[0])
        X = cho_solve((L, lower), B, overwrite_b=True)
        return X

    # Babylonian square root
    def sqrt(self, n):
        """
        Returns the square root of a positive number using Babylonian method.

        Args:
            n (float): The input number.

        Returns:
            float: The square root of n.
        """
        if n == 0:
            return 0
        x = n
        y = 1
        while x > y:
            x = (x + y) / 2
            y = n / x
        return x

    ##########################################################################
    # Methods for plotting distributions of data and loss/error over time
    ##########################################################################

    def plot_user_item_distribution(self):
        """
        Plots the distribution of the number of ratings per user and per
        item in a log-log scale.

        Returns: None.
        """
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
        plt.scatter(range(V.shape[0]), V, color=[0, 0, 0.69],
                    s=1.9, label='items')
        plt.scatter(range(U.shape[0]), U, color=[0, 0.69, 0],
                    s=1.9, label='users')
        plt.legend()
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Distribution of users and items.')

        if (self.small_dataset):
            plt.savefig("../report/small_user_item_dist.pdf",
                        bbox_inches="tight")
        else:
            plt.savefig("../report/full_user_item_dist.pdf",
                        bbox_inches="tight")

        plt.show()

    def plot_ratings_distribution(self):
        """
        Plots the distribution of ratings in two figures. The first figure
        shows the histogram of ratings where the ratings less than or equal
        to 1 are converted to 1. The second figure shows the histogram
        of original ratings.

        Returns: None.
        """
        converted_ratings = []
        for rating in self.ratings_u:
            if rating <= 1:
                converted_ratings.append(1)
            else:
                converted_ratings.append(int(rating))

        assert min(self.ratings_u) == 0.5

        plt.figure(figsize=(8, 5))
        plt.hist(converted_ratings, bins=20, range=(0.5, 5.5),
                 color="purple", edgecolor='black')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')

        if (self.small_dataset):
            plt.title('Rating distribution of the small dataset')
            plt.savefig("../report/small_ratings_dist_1.pdf",
                        bbox_inches="tight")
        else:
            plt.title('Rating distribution of the full dataset')
            plt.savefig("../report/full_ratings_dist_1.pdf",
                        bbox_inches="tight")

        plt.figure(figsize=(8, 5))
        plt.hist(self.ratings_u, bins=20, range=(0.5, 5.5),
                 color="purple", edgecolor='black')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')

        if (self.small_dataset):
            plt.title('Rating distribution of the small dataset')
            plt.savefig("../report/small_ratings_dist_2.pdf",
                        bbox_inches="tight")
        else:
            plt.title('Rating distribution of the full dataset')
            plt.savefig("../report/full_ratings_dist_2.pdf",
                        bbox_inches="tight")

        plt.show()

    def plot_loss(self):
        """
        Plots the loss function and the RMSE and MAE values over time
        during the training.

        Returns: None.

        Raises: Exception: If the model has not been trained or loaded.
        """
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        directory = "output/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "ml-25m/"
        directory += f"k = {self.k}, ll = {self.ll}, tau = {self.tau},"
        directory += f" alpha = {self.alpha}, beta = {self.beta},"
        directory += f" iters = {self.epoch}/"

        if (not os.path.isdir(directory)):
            os.makedirs(directory)

        plt.figure()
        plt.plot(range(1, len(self.loss_values) + 1),
                 np.array(self.loss_values).reshape(len(self.loss_values), ))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss function over time")
        plt.savefig(f"{directory}loss.pdf")

        plt.figure()
        plt.plot(range(1, len(self.rmse_values) + 1),
                 np.array(self.rmse_values).reshape(len(self.rmse_values), ),
                 label="RMSE")
        plt.plot(range(1, len(self.mae_values) + 1),
                 np.array(self.mae_values).reshape(len(self.mae_values), ),
                 label="MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Error value")
        plt.title("RMSE and MAE values over time")
        plt.legend()
        plt.savefig(f"{directory}rmse_mae.pdf")

        plt.show()

    def get_trait_vectors_with_names(self, spec_names: list):
        """
        Returns the trait vectors and the corresponding movie names for
        a given list of specific movie names.

        Args:
         - spec_names (list): The list of specific movie names.

        Returns: tuple: The tuple containing a list of trait vectors
                        and a list of corresponding movie names.
        """
        v_list = []
        name_list = []
        for spec in spec_names:
            for i, mov_name in enumerate(self.movie_names):
                if (spec in mov_name):
                    v_list.append(self.V[i])
                    name_list.append(mov_name)
        return v_list, name_list

    def plot_2D_vectors(self, spec_names: list):
        """
        Plots the trait vectors of movies with specific names in a 2D space.

        Args:
         - spec_names (list): The list of specific movie names.

        Returns: None.

        Raises: Exception: If the model has not been trained or loaded.
        """
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        directory = "output/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "ml-25m/"
        directory += f"k = {self.k}, ll = {self.ll}, tau = {self.tau},"
        directory += f" alpha = {self.alpha}, beta = {self.beta},"
        directory += f" iters = {self.epoch}/"

        if (not os.path.isdir(directory)):
            os.makedirs(directory)
        assert self.F[0].shape[0] == 2
        v_list, name_list = self.get_trait_vectors_with_names(
            spec_names=spec_names
        )

        plt.figure(figsize=(6, 6))
        plt.scatter(self.V[:, 1], self.V[:, 0], color="red",
                    label="Rest of movies", s=0.05)
        for v, name in zip(v_list, name_list):
            plt.scatter(x=v[1], y=v[0], color="blue", s=5)
            plt.text(x=v[1], y=v[0], s=name, fontsize=12)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title("2D embeddings of a selection of item trait vectors")

        if (self.small_dataset):
            plt.savefig(f"{directory}small_2D_item_vectors.pdf",
                        bbox_inches="tight")
        else:
            plt.savefig(f"{directory}full_2D_item_vectors.pdf",
                        bbox_inches="tight")

        plt.show()

    def plot_2D_genre_vectors(self):
        """
        Plots the trait vectors of genres in a 2D space.

        Returns: None.

        Raises:
         - Exception: If the model has not been trained or loaded.
         - AssertionError: If the self.k is not equal to 2.

        """
        if (not self.trained):
            raise Exception("Model has not been trained or loaded.")
        directory = "output/"
        if (self.small_dataset):
            directory += "small/"
        else:
            directory += "ml-25m/"
        directory += f"k = {self.k}, ll = {self.ll}, tau = {self.tau},"
        directory += f" alpha = {self.alpha}, beta = {self.beta},"
        directory += f" iters = {self.epoch}/"

        assert (self.F[0].shape[0] == 2) and (self.k == 2)

        plt.figure(figsize=(6, 6))
        for i, g in enumerate(self.unique_genres):
            plt.scatter(x=self.F[i][0], y=self.F[i][1], color="blue", s=7)
            plt.text(x=self.F[i][0], y=self.F[i][1], s=g, fontsize=9)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title("2D embeddings of the genre trait vectors")

        if (self.small_dataset):
            plt.savefig("../report/graphics/small_2D_genre_vectors.pdf",
                        bbox_inches="tight")
        else:
            plt.savefig("../report/graphics/full_2D_genre_vectors.pdf",
                        bbox_inches="tight")

        plt.show()


def main():
    model = MovieRecommender(small_dataset=True)
    model.train_model_als(max_epoch=25, save_model=True)
    model.plot_loss()


if __name__ == "__main__":
    main()
