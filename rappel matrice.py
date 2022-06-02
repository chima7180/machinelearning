import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]]) #definir une matrice

ab= a.shape # savoir m et n (m * n)

print(a)
print(ab)

ac = a.T #faire une transposé

print(ac)

b = np.ones((3, 2))
print(b)

print(a+b)

#print(a.dot(b)) car (3,2) par (3,2)

print(a.dot(b.T))
