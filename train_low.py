##############################################
#SCRIPT: train_low.py
#Autor: Rubén Revuelta Briz
#Definición: Creación y entrenamiento de una red neuronal con los datos proporcionados por BSH para
#            el trabajo de fin de grado del grado en Ingeniería Informática de la Universidad de Cantabria
##############################################

import tensorflow as tf 
import pandas as pd
from datetime import  datetime
import numpy as np
from tensorflow.contrib.layers import fully_connected
import math as mth
import argparse

def get_accuracy(predicciones, labels):
    '''Cálculo de la exactitud a partir de los resultados obtenidos.
                exactitud = (nº predicciones correctas) / (nº total de predicciones)
        Parámetros:
        predicciones -- predicciones llevadas a cabo por el modelo
        lavels -- valor real de los ejemplos
    '''
    predicciones_acertadas = 0
    for i in range(len(predicciones)):
        if predicciones[i] == labels[i]:
            predicciones_acertadas += 1
    
    return predicciones_acertadas/len(predicciones)

def get_recall(predicciones, labels):
    '''Cálculo de la exhaustividad a partir de los resultados obtenidos.
                exactitud = (vp)/(vp+fn)
        Parámetros:
        predicciones -- predicciones llevadas a cabo por el modelo
        lavels -- valor real de los ejemplos
    '''
    vp = 0
    fn = 0
    for i in range(len(predicciones)):
        if predicciones[i] == labels[i] and predicciones[i] == 1:
            vp += 1
        #El modelo ha precedido que una pieza es valida y  no lo es, falso negativo
        if predicciones[i] != labels[i] and labels[i] == 1:
            fn += 1

    return (vp)/(vp+fn)

def get_precision(predicciones, labels):
    '''Cálculo de la precisión a partir de los resultados obtenidos.
                exactitud = (vp)/(vp+fp)
        Parámetros:
        predicciones -- predicciones llevadas a cabo por el modelo
        lavels -- valor real de los ejemplos
    '''
    vp = 0
    fp = 0
    for i in range(len(predicciones)):
        if predicciones[i] == labels[i] and predicciones[i] == 1:
            vp += 1
        #El modelo ha predecido que una pieza es no v\'alida y la pieza es v\'alida
        if predicciones[i] != labels[i] and labels[i] == 0:
            fp += 1
    if fp == 0 and vp == 0:
        return 0

    return (vp)/(vp+fp)

def count_data(set):
    '''Cuenta el nº de datos disponibles de cada clase:
            - piezas válidas = negative_class
            - piezas no válidas = positive_class

        Parámetros:
        set -- conjunto de datos
    '''
    positive_class = 0
    negative_class = 0
    for i in range(len(set)):
        if set[i]==1:
            positive_class += 1
        else:
            negative_class += 1
    return positive_class, negative_class

#LLógica del control de parámetros del modelo
parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learningrate", help="Establece la tasa de aprendizaje", required=True, type=float)
parser.add_argument("-e", "--epochs", help="Numero de epocas del entrenamiento", required=True, type=int)
parser.add_argument("-bs", "--batchsize", help="Establece el tamaño de lote", required=True, type=int)
parser.add_argument("-nl", "--neuronslayer", help="Neuronas por capa, ejemplo de argumento para 3 capas: 128 64 32", required=True, type=int, nargs='+')
args = parser.parse_args()

learning_rate = args.learningrate
n_epochs = args.epochs
batch_size = args.batchsize
n_hidden = args.neuronslayer
num_capas = len(n_hidden)


#Carga de datos
training_labels = np.genfromtxt("over_training_labels.csv",delimiter=',',  dtype=int)
training_examples = np.genfromtxt("over_training_examples.csv",delimiter=',')

validation_labels = np.genfromtxt("validation_labels.csv",delimiter=',',  dtype=int)
validation_examples = np.genfromtxt("validation_examples.csv",delimiter=',')

#Lectura de pesos
training_weights = np.genfromtxt("training_weights.csv",delimiter=',')

num_piezas_no_valdias, num_piezas_validas = count_data(training_labels)

num_datos = len(training_labels)
batches_per_epoch = int(num_datos/batch_size)

n_inputs = 20
n_outputs = 2

#Placeholders de entrada
###############################################################
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
w = tf.placeholder(dtype=tf.float32, shape=(None), name="w")

#Placeholders auxiliares para la entrada de las medidas de evaluación
recall_tf =tf.placeholder(tf.float32, shape=(None), name="recall")
precision_tf =tf.placeholder(tf.float32, shape=(None), name="precision")
accuracy_tf =tf.placeholder(tf.float32, shape=(None), name="accuracy")

#Nombres de las capas
name_hidden = []
for i in range(num_capas):
    name_hidden.append("hidden" + str(i+1))
    print(name_hidden[i])

#FASE DE CONSTRUCCIÓN
#############################################################################################################################################################################################

#Definición de las dimensiones de la red
with tf.name_scope("dnn"):
    capas = []
    for i in range(num_capas):
        if i==0:
            capas.append(fully_connected(X, n_hidden[0], scope=name_hidden[0]))
        else:
            capas.append(fully_connected(capas[i-1], n_hidden[i-1], scope=name_hidden[i]))
    
    logits = fully_connected(capas[num_capas-1], n_outputs, scope="outputs", activation_fn=None)
    
    softmax = tf.nn.softmax(logits)

#Definición de la función de pérdida
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)  
    #xentropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y, weights=w)
    loss = tf.reduce_mean(xentropy, name="loss") 

#Entrenamiento basado en el algoritmo del gradiente descendente
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#Capa de evaluación de los resultados
with tf.name_scope("evaluation"):
    pred = tf.argmax(softmax,1)
    loss_summary = tf.summary.scalar('LOSS', loss)
    accuracy_summary = tf.summary.scalar("Validation Accuracy", accuracy_tf)
    recall_summary = tf.summary.scalar("Validation Recall", recall_tf)
    precision_summary = tf.summary.scalar("Validation Precision", precision_tf)

#FASE DE EJECUCIÓN
###############################################################

#Inicialización de la representación de datos en Tensorboard 
now = datetime.utcnow().strftime("%Y%m%d")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#Inicialización global de variables de tensorflow
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        #Por cada epoca recorremos todo el conjunto de entrenamiento dividido en lotes de tamaño batch_size
        for i in range(batches_per_epoch): 

            #Generación de lotes
            X_batch = training_examples[i*batch_size:(i+1)*batch_size]
            y_batch = training_labels[i*batch_size:(i+1)*batch_size]
            w_batch = training_weights[i*batch_size:(i+1)*batch_size]
            
            #sess.run(training_op, feed_dict={X: X_batch, y: y_batch, w: w_batch})
            #print_loss = loss.eval(feed_dict={X: X_batch, y: y_batch,  w: w_batch})
            
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            print_loss = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print("loss:", print_loss)

            step = epoch * (batches_per_epoch) + i
            
            prediccion = pred.eval(feed_dict={X: validation_examples, y: validation_labels})
            
            #Cálculo de las medidas de evaluación aplicadas sobre datos de validación
            summary_loss = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
            #summary_loss = loss_summary.eval(feed_dict={X: X_batch, y: y_batch,  w: w_batch})
            summary_accuracy = accuracy_summary.eval(feed_dict={accuracy_tf: get_accuracy(prediccion, validation_labels)})  
            summary_recall = recall_summary.eval(feed_dict={recall_tf: get_recall(prediccion, validation_labels) })
            summary_precision = precision_summary.eval(feed_dict={precision_tf: get_precision(prediccion, validation_labels) })

            #Escritura de las medidas en tensorboard
            writer.add_summary(summary_loss, step)
            writer.add_summary(summary_accuracy, step)
            writer.add_summary(summary_recall, step)
            writer.add_summary(summary_precision, step)
    
    save_path = saver.save(sess, "./my_model_final.ckpt") 
   





