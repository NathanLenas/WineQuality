import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, metrics

import pandas as pd

import random

from copy import deepcopy
from io import StringIO

def getModel():
    model = None
    try:
        model = tf.keras.models.load_model('../data/model')
    except:
        train()
        model = tf.keras.models.load_model('../data/model')
        
    return model


def train():
    data = pd.read_csv("../data/Wines.csv")

    X = deepcopy(data)
    X = X.sample(frac = 1)
    print(data)
    print(X)
    Y = deepcopy(X["quality"])

    del X["quality"]
    del X["Id"]

    # Normalisation

    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]

    Y_train = Y[:int(0.8 * len(Y))]
    Y_test = Y[int(0.8 * len(Y)):]

    inputs = Input(11)

    h1 = Dense(64, activation="relu")(inputs)
    h2 = Dense(64, activation="relu")(h1)
    h3 = Dense(64, activation="relu")(h2)
    outputs = Dense(10, activation="softmax")(h3)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=100, 
                        validation_data=(X_test, Y_test))
    
    # Save a model
    model.save('../data/model')


def description():
    model = getModel()
    
    # Create a StringIO object to redirect the console output
    buffer = StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    
    # Get the output from the buffer and return it
    desc = buffer.getvalue()
    return desc
   



def predire(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol):
    model = getModel()

    prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
    i = 0
    indice = 0
    max = 0
    for x in prediction[0]:
        i = i + 1
        if(x > max):
            max = x
            indice = i        
    return indice

# Some libraries
from scipy import *
from math import *
from matplotlib.pyplot import *
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from functools import *
import sys
from statistics import *

from numpy import *

DIM = 11    # problem dimension
# domain of variation definition
INF = [ 3, 0, 0, 0,   0,  0,   0, 0,  0, 0,  7]
SUP = [18, 2, 1, 6, 0.3, 70, 350, 1, 7, 1, 17]

Nb_cycles = 10
Nb_particle = 3
# usual params
psi,cmax = (0.8, 1.62)

def dispRes(best):
    print("point = {}".format(best['pos']))
    print("eval = {}".format(best['fit']))

def eval(sol):
    model = tf.keras.models.load_model('../data/model')
    prediction = model.predict([sol], verbose=0)
    return(prediction[0][9])

def initOne(dim,inf,sup):
    pos = [random.uniform(inf[i], sup[i]) for i in range(dim)]
    fit = eval(pos)
    return {'vit':[0]*dim, 'pos':pos, 'fit':fit, 'bestpos':pos[:], 'bestfit':fit, 'bestvois':[]}


# Init of the population (swarm)
def initSwarm(nb,dim,inf,sup):
    return [initOne(dim,inf,sup) for i in range(nb)]

# Return the particle with the best fitness
def maxParticle(p1,p2):
    if (p1["fit"] > p2["fit"]):
        return p1 
    else:
        return p2

# Returns a copy of the particle with the best fitness in the population
def getBest(swarm):
    return dict(reduce(lambda acc, e: maxParticle(acc,e),swarm[1:],swarm[0]))

# function to stop at boundaries of the search space
def bornage(val, inf, sup):
    return min (sup, max (inf, val))

# Update information for the particles of the population (swarm)
def update(particle,bestParticle):
    nv = dict(particle)
    if(particle["fit"] > particle["bestfit"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestfit'] = particle["fit"]
    nv['bestvois'] = bestParticle["bestpos"][:]
    return nv

# Calculate the velocity and move a paticule
def move(particle,dim):
    global ksi,c1,c2,psi,cmax,INF,SUP
    nv = dict(particle)

    #velocity = [0]*dim
    #for i in range(dim):
    #    velocity[i] = (particle["vit"][i]*psi + \
    #    cmax*random.uniform()*(particle["bestpos"][i] - particle["pos"][i]) + \
    #    cmax*random.uniform()*(particle["bestvois"][i] - particle["pos"][i])) 
    velocity = [(particle["vit"][i]*psi + \
        cmax*random.uniform()*(particle["bestpos"][i] - particle["pos"][i]) + \
        cmax*random.uniform()*(particle["bestvois"][i] - particle["pos"][i])) \
        for i in range(dim)]
    
    
    #position = [0]*dim
    #for i in range(dim):
    #    position[i] = bornage(particle["pos"][i] + velocity[i], INF[i], SUP[i])
    position = [bornage(particle["pos"][i] + velocity[i], INF[i], SUP[i]) for i in range(DIM)]
    
    nv['vit'] = velocity
    nv['pos'] = position
    nv['fit'] = eval(position)

    return nv

def parfait():

    # MAIN LOOP

    # initialization of the population
    swarm = initSwarm(Nb_particle,DIM,INF,SUP)
    # initialization of the best solution
    best = getBest(swarm)
    best_cycle = best

    for i in range(Nb_cycles):
        #Update informations
        swarm = [update(e,best_cycle) for e in swarm]
        # velocity calculations and displacement
        swarm = [move(e,DIM) for e in swarm]
        # Update of the best solution
        best_cycle = getBest(swarm)
        if (best_cycle["bestfit"] < best["bestfit"]):
            best = best_cycle
        
    # END, displaying results
    return (best_cycle["bestpos"])