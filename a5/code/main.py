import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2, kernel_poly
from pca import PCA, AlternativePCA, RobustPCA


def load_dataset(filename):
    with open(os.path.join("..", "data", filename), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        dataset = load_dataset("nonLinearData.pkl")
        X = dataset["X"]
        y = dataset["y"]

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")

        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset("nonLinearData.pkl")
        X = dataset["X"]
        y = dataset["y"]

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

        # YOUR CODE HERE
        # kernel logistic regression with a RBFs kernel
        RBFsModel = kernelLogRegL2(lammy=0.01, sigma=0.5)
        RBFsModel.fit(Xtrain, ytrain)
        print(
            "Training error RBFs Kernel %.3f"
            % np.mean(RBFsModel.predict(Xtrain) != ytrain)
        )
        print(
            "Validation error RBFs Kernel %.3f"
            % np.mean(RBFsModel.predict(Xtest) != ytest)
        )

        utils.plotClassifier(RBFsModel, Xtrain, ytrain)
        utils.savefig("RBFs_kernel.png")

        # kernel logistic regression with a polynomial kernel
        polyModel = kernelLogRegL2(lammy=0.01, kernel_fun=kernel_poly)
        polyModel.fit(Xtrain, ytrain)
        print(
            "Training error polynomial kernel %.3f"
            % np.mean(polyModel.predict(Xtrain) != ytrain)
        )
        print(
            "Validation error polynomial kernel %.3f"
            % np.mean(polyModel.predict(Xtest) != ytest)
        )
        utils.plotClassifier(polyModel, Xtrain, ytrain)
        utils.savefig("polynomial_kernel.png")

    elif question == "1.2":
        dataset = load_dataset("nonLinearData.pkl")
        X = dataset["X"]
        y = dataset["y"]
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
        # YOUR CODE HERE
        # Data for three-dimensional scattered points
        sigmaRange = range(-2, 3)
        lamRange = range(-4, 1)
        zdata_train = np.zeros((len(sigmaRange), len(lamRange)))
        zdata_valid = np.zeros((len(sigmaRange), len(lamRange)))
        xdata = np.zeros((len(sigmaRange), len(lamRange)))
        ydata = np.zeros((len(sigmaRange), len(lamRange)))
        bestLamVa = -1
        bestLamTr = -1
        bestSigmaVa = -1
        bestSigmaTr = -1
        minVal = np.infty
        minTr = np.infty
        for i in range(len(sigmaRange)):
            for j in range(len(lamRange)):
                sigma = 10 ** sigmaRange[i]
                lammy = 10 ** lamRange[j]
                model = kernelLogRegL2(lammy=lammy, sigma=sigma)
                model.fit(Xtrain, ytrain)
                train_er = np.mean(model.predict(Xtrain) != ytrain)
                test_er = np.mean(model.predict(Xtest) != ytest)
                zdata_train[i, j] = train_er
                zdata_valid[i, j] = test_er
                xdata[i, j] = sigmaRange[i]
                ydata[i, j] = lamRange[j]
                if train_er < minTr:
                    bestLamTr = lammy
                    bestSigmaTr = sigma
                    minTr = train_er
                if test_er < minVal:
                    bestLamVa = lammy
                    bestSigmaVa = sigma
                    minVal = test_er

        plt.scatter(xdata, ydata, s=20, c=zdata_train, cmap="Greys_r")
        plt.xlabel("sigma (10**)")
        plt.ylabel("lambda (10**)")
        fname = "q1-2-train.pdf"
        path = os.path.join("..", "figs", fname)
        plt.savefig(path)
        plt.scatter(xdata, ydata, s=20, c=zdata_valid, cmap="Greys_r")
        plt.xlabel("sigma (10**)")
        plt.ylabel("lambda (10**)")
        fname = "q1-2-valid.pdf"
        path = os.path.join("..", "figs", fname)
        plt.savefig(path)
        print(f"best sigma for training: {bestSigmaTr}")
        print(f"best lambda for training: {bestLamTr}")
        print(f"best sigma for validation: {bestSigmaVa}")
        print(f"best lambda for validation: {bestLamVa}")

    elif question == "4.1":
        X = load_dataset("highway.pkl")["X"].astype(float) / 255
        n, d = X.shape
        print(n, d)
        h, w = 64, 64  # height and width of each image

        k = 5  # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2, 3)
        for i in range(10):
            ax[0, 0].set_title("$X$")
            ax[0, 0].imshow(X[i].reshape(h, w).T, cmap="gray")

            ax[0, 1].set_title("$\hat{X}$ (L2)")
            ax[0, 1].imshow(Xhat_pca[i].reshape(h, w).T, cmap="gray")

            ax[0, 2].set_title("$|x_i-\hat{x_i}|$>threshold (L2)")
            ax[0, 2].imshow(
                (np.abs(X[i] - Xhat_pca[i]) < threshold).reshape(h, w).T,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            ax[1, 0].set_title("$X$")
            ax[1, 0].imshow(X[i].reshape(h, w).T, cmap="gray")

            ax[1, 1].set_title("$\hat{X}$ (L1)")
            ax[1, 1].imshow(Xhat_robust[i].reshape(h, w).T, cmap="gray")

            ax[1, 2].set_title("$|x_i-\hat{x_i}|$>threshold (L1)")
            ax[1, 2].imshow(
                (np.abs(X[i] - Xhat_robust[i]) < threshold).reshape(h, w).T,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            utils.savefig("highway_{:03d}.jpg".format(i))

    else:
        print("Unknown question: %s" % question)

