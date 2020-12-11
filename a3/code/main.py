# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils


url_amazon = "https://www.amazon.com/dp/%s"


def load_dataset(filename):
    with open(os.path.join("..", "data", filename), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f, names=("user", "item", "rating", "timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings) / (n * d))

        (
            X,
            user_mapper,
            item_mapper,
            user_inverse_mapper,
            item_inverse_mapper,
            user_ind,
            item_ind,
        ) = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f, names=("user", "item", "rating", "timestamp"))
        (
            X,
            user_mapper,
            item_mapper,
            user_inverse_mapper,
            item_inverse_mapper,
            user_ind,
            item_ind,
        ) = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        # YOUR CODE HERE FOR Q1.1.1
        rating_sum = np.sum(X, axis=0)
        mx_idx = np.argmax(rating_sum)
        print(item_inverse_mapper[mx_idx])
        print(np.max(rating_sum))

        # YOUR CODE HERE FOR Q1.1.2
        u = np.sum(X != 0, axis=1)
        mx_idx = np.argmax(u)
        print(user_inverse_mapper[mx_idx])
        print(np.max(u))

        # YOUR CODE HERE FOR Q1.1.3
        # 1.The number of ratings per user

        # 2.  The number of ratings per item

        # 3.  The ratings themselves

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f, names=("user", "item", "rating", "timestamp"))
        (
            X,
            user_mapper,
            item_mapper,
            user_inverse_mapper,
            item_inverse_mapper,
            user_ind,
            item_ind,
        ) = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:, grill_brush_ind]
        Z = np.transpose(X)

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        model = NearestNeighbors()
        model.fit(Z)
        neighbors = model.kneighbors(X=np.transpose(grill_brush_vec), n_neighbors=6)
        indices = neighbors[1]
        for i in range(6):
            print(item_inverse_mapper[indices[0, i]])
            pass

        Z_norm = normalize(Z)
        model2 = NearestNeighbors()
        model2.fit(Z_norm)
        neighbors = model2.kneighbors(X=np.transpose(grill_brush_vec), n_neighbors=6)
        indices = neighbors[1]
        for i in range(6):
            print(item_inverse_mapper[indices[0, i]])
            pass

        model3 = NearestNeighbors(metric="cosine")
        model3.fit(Z)
        neighbors = model3.kneighbors(X=np.transpose(grill_brush_vec), n_neighbors=6)
        indices = neighbors[1]
        for i in range(6):
            print(item_inverse_mapper[indices[0, i]])
            pass

        # YOUR CODE HERE FOR Q1.3

    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data["X"]
        y = data["y"]

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X, y)
        print(model.w)

        utils.test_and_plot(
            model, X, y, title="Least Squares", filename="least_squares_outliers.pdf"
        )

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data["X"]
        y = data["y"]

        # YOUR CODE HERE
        print(X.shape)
        print(y.shape)
        v = np.ones(500)
        v[400:] = 0.1
        model = linear_model.WeightedLeastSquares()
        model.fit(X, y, v)
        print(model.w)

        utils.test_and_plot(
            model,
            X,
            y,
            title="Weighted Least Squares",
            filename="weighted_least_squares_outliers.pdf",
        )

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data["X"]
        y = data["y"]
        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X, y)
        print(model.w)

        utils.test_and_plot(
            model,
            X,
            y,
            title="Robust (L1) Linear Regression",
            filename="least_squares_robust.pdf",
        )

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data["X"]
        y = data["y"]
        Xtest = data["Xtest"]
        ytest = data["ytest"]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X, y)

        utils.test_and_plot(
            model,
            X,
            y,
            Xtest,
            ytest,
            title="Least Squares, no bias",
            filename="least_squares_no_bias.pdf",
        )

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data["X"]
        y = data["y"]
        Xtest = data["Xtest"]
        ytest = data["ytest"]

        # YOUR CODE HERE
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(
            model,
            X,
            y,
            Xtest,
            ytest,
            title="Least Squares, with bias",
            filename="least_squares_with_bias.pdf",
        )

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data["X"]
        y = data["y"]
        Xtest = data["Xtest"]
        ytest = data["ytest"]

        for p in range(11):
            print("p=%d" % p)

            # YOUR CODE HERE
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            utils.test_and_plot(
                model,
                X,
                y,
                Xtest,
                ytest,
                title="Least Squares, with bias",
                filename="least_squares_with_bias.pdf",
            )

    else:
        print("Unknown question: %s" % question)
