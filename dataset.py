##############################################
#SCRIPT: dataset.py
#Autor: Rubén Revuelta Briz
#Definición: Pre-procesado del conjunto 
#            de datos de BSH para su posterior 
#            entrenamiento.
##############################################

import pandas as pd 
import numpy as np 
import os
import glob
import math as mth
import tensorflow as tf 
import itertools
from matplotlib import pyplot as plt
from sklearn import metrics
from IPython import display

#Función aplicada al ATS_ERROR_CODE de los datos
def f(x):
        """Estable el valor de la etiqueta a 0 o 1.

                 0 -> pieza válida
                 1 -> pieza no válida

        Parámetros:
        x -- Valor inicial de la etiqueta.
        """
        if x == 0:
                return int(0) #Pieza correcta
        else:
                return int(1) #Pieza incorrecta

def preprocess_features(bsh_dataframe):
        """Elimina el ATS_ERROR_CODE del DataFrame original.

        Parámetros:
        bsh_dataframe -- DataFrame de BSH
        """
        features = bsh_dataframe.drop('ATS_ERROR_CODE', 1)

        return features

def preprocess_targets(bsh_dataframe):
        """Extrae el atributo correspondiente a las etiquetas del DataFrame.

        Parámetros:
        bsh_dataframe -- DataFrame de BSH
        """
        targets =  pd.Series(bsh_dataframe['ATS_ERROR_CODE'].values, 
                                                index = bsh_dataframe.index)
        targets = targets.apply(f)

        return targets

def load_dataFrame():
        """
        Extrae los datos almacenados en los ficheros .xls 
        y los introduce en un DataFrame de pandas.
        """
        enero_17=pd.read_excel('./datos/Prod_01-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU",header=12)
        febrero_17= pd.read_excel('./datos/Prod_02-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        marzo_17= pd.read_excel('./datos/Prod_03-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        abril_17= pd.read_excel('./datos/Prod_04-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        mayo_17= pd.read_excel('./datos/Prod_05-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        junio_17= pd.read_excel('./datos/Prod_06-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        julio_17= pd.read_excel('./datos/Prod_07-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        agosto_17= pd.read_excel('./datos/Prod_08-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        septiembre_17= pd.read_excel('./datos/Prod_09-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        octubre_17= pd.read_excel('./datos/Prod_10-17.xls'
                , sheet_name='Prod_Filter', usecols="B:BU")
        noviembre_17= pd.read_excel('./datos/Prod_11-17.xls'   
                , sheet_name=1, usecols="B:BU")
        diciembre_17= pd.read_excel('./datos/Prod_12-17.xls'
                , sheet_name=1, usecols="B:BU")

        enero_18=pd.read_excel('./datos/Prod_01-18.xls'
                , sheet_name='Prod', usecols="B:BU")
        febrero_18= pd.read_excel('./datos/Prod_02-18.xls'
                , sheet_name='Prod', usecols="B:BU")
        marzo_18= pd.read_excel('./datos/Prod_03-18.xls'
                , sheet_name='Prod', usecols="B:BU")
        abril_18= pd.read_excel('./datos/Prod_04-18.xls'
                , sheet_name='Prod', usecols="B:BU")
        mayo_18= pd.read_excel('./datos/Prod_05-18.xls'
                , sheet_name='Prod', usecols="B:BU")

        #Concatenación de los datos obtenidos de los 12 ficheros 
        dataframe= pd.concat([enero_17, febrero_17,marzo_17,abril_17,mayo_17,
                        junio_17,julio_17,agosto_17,septiembre_17,octubre_17,
                        noviembre_17,diciembre_17,enero_18, febrero_18, 
                        marzo_18, abril_18, mayo_18], axis=0)

        dataframe= dataframe.reset_index(drop=True)
        
        return dataframe


###########################################################
# TRATAMIENTO DE LOS DATOS
##########################################################

#Carga de los datos en los ficheros .xls en un dataframe
bsh_dataframe = load_dataFrame()


#Medidas para el analisis del dataFrame
display.display(bsh_dataframe.describe())

#Filtro de datos
bsh_dataframe= bsh_dataframe[((bsh_dataframe['MATNR']==9001073660) & (bsh_dataframe['TURN_COUNTS']==1)) 
                                & (((bsh_dataframe['ATS_ERROR_CODE']>=50000) & (bsh_dataframe['ATS_ERROR_CODE']<=59999)) | (bsh_dataframe['ATS_ERROR_CODE']==0))
                                & (bsh_dataframe['ST635_RIVET_NOTCH_STRO']!=99)] 

#Filtro de los atributos que serán utilizados para el entrenamiento
bsh_dataframe = bsh_dataframe[['ATS_ERROR_CODE','ST265_RING_PLA_CLIN_FORC',
                'ST265_RING_PLA_CLIN_STRO' ,'ST290_SPINDLE_FORC', 
                'ST290_SPINDLE_STRO','ST325_LOW_COV_PLA_FORC', 
                'ST325_LOW_COV_PLA_STRO','ST345_LOW_COV_PLA_FORC_1', 
                'ST345_LOW_COV_PLA_FORC_2','ST345_LOW_COV_PLA_FORC_3', 
                'ST345_LOW_COV_PLA_STRO_1','ST345_LOW_COV_PLA_STRO_2', 
                'ST345_LOW_COV_PLA_STRO_3','ST380_NOTCH_PIN_FORC' ,
                'ST380_NOTCH_PIN_STRO', 'ST415_INPUT_DOWEL_FORC',
                'ST415_INPUT_DOWEL_STRO' ,'ST455_TAPCAP_SCREW_ANG_1',
                'ST455_TAPCAP_SCREW_ANG_2' ,'ST455_TAPCAP_SCREW_TOR_1',
                'ST455_TAPCAP_SCREW_TOR_2']]

lista_columnas = bsh_dataframe.columns.values

#Filto que elimina los ejemplos que contenga atributos a 0
for columna in lista_columnas:
        if columna != 'ATS_ERROR_CODE':
                bsh_dataframe = bsh_dataframe[bsh_dataframe[columna]!=0]

bsh_dataframe_labels_error = bsh_dataframe[bsh_dataframe['ATS_ERROR_CODE']!=0]
bsh_dataframe_labels_error.to_csv("bsh_dataframe_labels_error.csv")

bsh_dataframe_labels_correcta = bsh_dataframe[bsh_dataframe['ATS_ERROR_CODE']==0]
bsh_dataframe_labels_correcta.to_csv("bsh_dataframe_labels_correcta.csv")

num_ejemplos_error = bsh_dataframe_labels_error.shape[0]
num_ejemplos_correctos =bsh_dataframe_labels_correcta.shape[0]
print("Num piezas erroneas: " + str(num_ejemplos_error))
print("Num piezas correctas: " + str(num_ejemplos_correctos))

#Cálculo del ratio para la aplicación de la técnica cost sensitive learning
ratio = num_ejemplos_error / (num_ejemplos_correctos + num_ejemplos_error)

#Randomnización de los datos
bsh_dataframe = bsh_dataframe.reindex(np.random.permutation(bsh_dataframe.index))

bsh_dataframe.to_csv("bsh_dataframe_no_normalizado.csv")

print(bsh_dataframe.min())
print(bsh_dataframe.max())

#Normalización de los datos
bsh_dataframe = (bsh_dataframe - bsh_dataframe.min()) / (bsh_dataframe.max() - bsh_dataframe.min())

#Calculo de los pesos en función del ratio
bsh_dataframe['WEIGHT']=np.array([1-ratio if  fila[1][0]!= 0 else ratio for fila in bsh_dataframe.iterrows()])
weights = bsh_dataframe['WEIGHT']
#Eliminación de la columna correspondiente a los pesos del dataframe principal
bsh_dataframe = bsh_dataframe.drop('WEIGHT', 1)

#Escritura de los pesos en un fichero .csv
weights.to_csv("weights.csv", index=False, header = False)

#Comprobación min y max de los datos ya normalizados, permite ver valores normalizados atípicos
print(bsh_dataframe.min())
print(bsh_dataframe.max())

#Modificación de las etiquetas (0 correcta, 1 incorrecta)
bsh_dataframe['ATS_ERROR_CODE']= bsh_dataframe['ATS_ERROR_CODE'].apply(f)

#Conjunto completo de datos
bsh_dataframe.to_csv("bsh_dataframe.csv", index=False, header=False)


examples = bsh_dataframe.drop('ATS_ERROR_CODE', 1)
labels = bsh_dataframe['ATS_ERROR_CODE']

examples.to_csv("examples.csv", index=False, header=False)
labels.to_csv("labels.csv", index=False, header=False)

num_datos = examples.shape[0]
#El número de ejemplos destinados a validación se corresponde con el 20% de los ejemplos total que forman el dataset
num_validacion = mth.floor(num_datos*0.2)

#Separación del conjunto de datos en datos de entrenamiento y validación
training_examples = examples.head(num_datos-num_validacion)
training_labels = labels.head(num_datos-num_validacion)

#únicamente se utilizan pesos con los ejemplos de validacón
training_weights = weights.head(num_datos-num_validacion)

validation_examples = examples.tail(num_validacion)
validation_labels = labels.tail(num_validacion)


weights.to_csv("training_weights.csv", index=False, header = False)

training_examples.to_csv("training_examples.csv", index=False, header=False)
training_labels.to_csv("training_labels.csv", index=False, header=False)

validation_examples.to_csv("validation_examples.csv", index=False, header=False)
validation_labels.to_csv("validation_labels.csv", index=False, header=False)


#UNDER-SAMPLING
##########################################################################

#Randomnización de los ejemplos correctos
bsh_dataframe_labels_correcta = bsh_dataframe_labels_correcta.reindex(np.random.permutation(bsh_dataframe_labels_correcta.index))

#Nos quedamos con un 15% más del número de ejemplos erróneos
under_correctas = bsh_dataframe_labels_correcta.head(num_ejemplos_error + mth.floor(0.15*num_ejemplos_error))

under_sampling_bsh = pd.concat([under_correctas, bsh_dataframe_labels_error], axis=0)

#Randomnización de los datos
under_sampling_bsh =under_sampling_bsh.reindex(np.random.permutation(under_sampling_bsh.index))

under_sampling_bsh.to_csv("under_sampling.csv", index=False, header=False)

#Normalización de los datos
under_sampling_bsh = (under_sampling_bsh - under_sampling_bsh.min()) / (under_sampling_bsh.max() - under_sampling_bsh.min())

#Modificación de las etiquetas (0 correcta, 1 incorrecta)
under_sampling_bsh['ATS_ERROR_CODE']= under_sampling_bsh['ATS_ERROR_CODE'].apply(f)

examples = under_sampling_bsh.drop('ATS_ERROR_CODE', 1)
labels = under_sampling_bsh['ATS_ERROR_CODE']

num_datos = examples.shape[0]
#El número de ejemplos destinados a validación se corresponde con el 20% de los ejemplos total que forman el dataset
num_validacion = mth.floor(num_datos*0.2)

under_training_examples = examples.head(num_datos-num_validacion)
under_training_labels = labels.head(num_datos-num_validacion)

under_validation_examples = examples.tail(num_validacion)
under_validation_labels = labels.tail(num_validacion)

#Escritura de los datos en ficheros .csv para su posterior lectura
under_training_examples.to_csv("under_training_examples.csv", index=False, header=False)
under_training_labels.to_csv("under_training_labels.csv", index=False, header=False)

under_validation_examples.to_csv("under_validation_examples.csv", index=False, header=False)
under_validation_labels.to_csv("under_validation_labels.csv", index=False, header=False)

#OVERSAMPLING
##########################################################################
names_examples = ['ST265_RING_PLA_CLIN_FORC',
                'ST265_RING_PLA_CLIN_STRO' ,'ST290_SPINDLE_FORC', 'ST290_SPINDLE_STRO',
                'ST325_LOW_COV_PLA_FORC', 'ST325_LOW_COV_PLA_STRO',
                'ST345_LOW_COV_PLA_FORC_1', 'ST345_LOW_COV_PLA_FORC_2',
                'ST345_LOW_COV_PLA_FORC_3', 'ST345_LOW_COV_PLA_STRO_1',
                'ST345_LOW_COV_PLA_STRO_2', 'ST345_LOW_COV_PLA_STRO_3',
                'ST380_NOTCH_PIN_FORC' ,'ST380_NOTCH_PIN_STRO', 'ST415_INPUT_DOWEL_FORC',
                'ST415_INPUT_DOWEL_STRO' ,'ST455_TAPCAP_SCREW_ANG_1',
                'ST455_TAPCAP_SCREW_ANG_2' ,'ST455_TAPCAP_SCREW_TOR_1',
                'ST455_TAPCAP_SCREW_TOR_2']
names_labels = ["ATS_ERROR_CODE"]

training_labels = pd.read_csv("training_labels.csv", index_col=False, names = names_labels)
training_examples = pd.read_csv("training_examples.csv", index_col=False, names = names_examples)

#Concatenación de las etiquetas con sus respectivos atributos
bsh_dataframe = pd.concat([training_examples,training_labels],axis=1)

#Selección de ejemplos, válidos y no válidos
bsh_dataframe_labels_error = bsh_dataframe[bsh_dataframe['ATS_ERROR_CODE']!=0]
bsh_dataframe_labels_correcta = bsh_dataframe[bsh_dataframe['ATS_ERROR_CODE']==0]

num_datos = bsh_dataframe.shape[0]
num_ejemplos_error = bsh_dataframe_labels_error.shape[0]

#Randomnización de los ejemplos correctos
bsh_dataframe_labels_error = bsh_dataframe_labels_error.reindex(np.random.permutation(bsh_dataframe_labels_error.index))

#El número de ejemplos creados de la clase positiva es igual al 75% de los datos menos el número de ejemplos erroneos ya existentes
num_over = int(0.75*num_datos - num_ejemplos_error)

over_labels_error = bsh_dataframe_labels_error

#se eligen "num_over" ejemplos de forma aleatoria y se insertan de nuevo al dataframe
over_labels_error = over_labels_error.append(bsh_dataframe_labels_error.sample(n=num_over, replace=True))

over_sampling_bsh = pd.concat([bsh_dataframe_labels_correcta, over_labels_error], axis=0)
over_sampling_bsh = over_sampling_bsh.reset_index(drop=True)

#Randomnización de los datos
over_sampling_bsh =over_sampling_bsh.reindex(np.random.permutation(over_sampling_bsh.index))

over_sampling_bsh.to_csv("over_sampling.csv", index=False, header=False)

#Normalización de los datos
over_sampling_bsh = (over_sampling_bsh - over_sampling_bsh.min()) / (over_sampling_bsh.max() - over_sampling_bsh.min())

#Modificación de las etiquetas (0 correcta, 1 incorrecta)
over_sampling_bsh['ATS_ERROR_CODE']= over_sampling_bsh['ATS_ERROR_CODE'].apply(f)

examples = over_sampling_bsh.drop('ATS_ERROR_CODE', 1)
labels = over_sampling_bsh['ATS_ERROR_CODE']


#Escritura en ficheros .csv para su posterior lectura
examples.to_csv("over_training_examples.csv", index=False, header=False)
labels.to_csv("over_training_labels.csv", index=False, header=False)
