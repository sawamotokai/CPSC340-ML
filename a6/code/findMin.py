import numpy as np
from numpy.linalg import norm


def SGD(funObj, w, maxEpoch, batchsize, *args, verbose=0):
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    N = args[0].shape[0]
    # Evaluate the initial function value and gradient
    f, g = funObj(w, *args)
    funEvals = 1

    alpha = 0.001
    iterPerEpoch = N // batchsize

    for epoch in range(1, maxEpoch + 1):
        # Line-search using quadratic interpolation to
        # find an acceptable value of alpha
        gg = g.T.dot(g)

        for _ in range(iterPerEpoch):
            indices = np.random.choice(np.arange(N), batchsize)
            Xtrain = args[0][indices]
            ytrain = args[1][indices]
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, Xtrain, ytrain)
            # Update parameters/function/gradient
            w = w_new
            f = f_new
            g = g_new

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (epoch, f_new))

    return w, f


def findMin(funObj, w, maxEvals, *args, verbose=0):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w, *args)
    funEvals = 1

    alpha = 1.0
    while True:
        # Line-search using quadratic interpolation to
        # find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1
            if f_new <= f - gamma * alpha * gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha ** 2) * gg / (2.0 * (f_new - f + alpha * gg))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.0

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float("inf"))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f