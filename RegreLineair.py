import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


'''
1.Génération d'un Dataset
'''
x,y =make_regression(n_samples=100, n_features=1,noise=10)
plt.scatter(x, y)
print(x.shape)
y = y.reshape(y.shape[0], 1) #si y n'est pas complet (100, ) écrire la ligne ci contre
print(y.shape)

#matrice x
X = np.hstack((x, np.ones(x.shape)))
print(X)

#Matrice Theta
theta = np.random.randn(2, 1)
# print(theta)

'''
2.Modèle linaire
'''

def model (X, theta):
    return X.dot(theta)

#affichage de la fonction
plt.plot(x, model(X, theta))

'''
3.Fonction cout
'''

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta)-y)**2)

#affichage de la fonction
cost_function(X, y, theta)


'''
4.Descente de Gradient
'''

def grad(X, y, theta):
    m=len(y)
    return 1/m * X.T.dot(model(X,theta)-y)

def gradient_descnet(X, y, theta, learning_rate, n_iterations):

    cost_history = np.zeros(n_iterations)
    for i in range (0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history

'''
Machine learning !!
'''

theta_final, cost_history = gradient_descnet(X, y, theta, learning_rate=0.01, n_iterations=1000)
print(theta_final)

predictions = model(X, theta_final)
plt.scatter(x , y)
plt.plot(x, predictions, c='r')


'''
courbe d'apprentissage
ajout de cost history dans la fonction gradient_descent
'''

plt.plot(range(1000), cost_history)
#a note que au dela de 400 faire continuer de faire touner la machine est usless

'''
Coef de determination 
'''

def coef_determination(y, pred):
    u = ((y-pred)**2).sum()
    v = ((y-y.mean())**2).sum()
    return 1 - u/v

coef_determination(y, predictions)
