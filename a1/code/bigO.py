import numpy as np


def func1(N):
    for i in range(N):
        print("Hello!")


def func2(N):
    x = np.zeros(N)
    x += 1000
    return x


def func3(N):
    x = np.zeros(1000)
    x = x * N
    return x


def func4(N):
    x = 0
    for i in range(N):
        for j in range(i, N):
            x += i * j
    return x


# func1
# O(N): It has a for loop that loops N times
#
# func2
# O(N): It initializes an array of length N with 0's and it then adds 1000 to all the elements; O(N+N) => O(N)
#
# func3
# O(1):
#     It initializes an array of length 3000, which is independent of N and it then multiplies N to all the 3000 elements; O(3000) => O(1)
#
# func4
# O(N^2): It has a nested loop of N*N and addition can be considered as constant; O(N^2*1) => O(N^2)

