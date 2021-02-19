# skforecast: Forecasting series temporales con Scikitlearn

## Instalación
<br>

<code> pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master </code>

## Introducción
<br>

Una [serie temporal](https://es.wikipedia.org/wiki/Serie_temporal) (*time series*) es una sucesión de datos ordenados cronológicamente, espaciados a intervalos iguales o desiguales. 

El proceso de [*Forecasting*](https://en.wikipedia.org/wiki/Forecasting) consiste en predecir el valor futuro de una serie temporal, bien modelando la serie temporal únicamente en función de su comportamiento pasado (autorregresivo) o empleando otras variables externas a la serie temporal.

A lo largo de este documento, se describe cómo **skforecast**  permite aplicar modelos de regresión de **Scikit learn**  a problemas de *forecasting* sobre series temporales.
<br><br>

## Forecasting
<br>

Cuando se trabaja con series temporales, raramente se necesita predecir el siguiente elemento de la serie ($t_{+1}$), sino todo un intervalo futuro o un punto alejado en el tiempo ($t_{+n}$). 

Dado que, para predecir el momento $t_{+2}$ se necesita el valor de $t_{+1}$, y $t_{+1}$ se desconoce, es necesario hacer predicciones recursivas en las que, cada nueva predicción, se basa en la predicción anterior. A este proceso se le conoce *recursive forecasting* o *multi-step forecasting* y es la principal diferencia respecto a los problemas de regresión convencionales.

<p><img src="./images/forecasting_multi-step.gif" alt="forecasting-python" title="forecasting-python"></p>

## Forecasting autorregresivo con scikit learn
<br>

La principal adaptación que se necesita hacer para aplicar modelos de [scikit learn](https://www.cienciadedatos.net/documentos/py06_machine_learning_python_scikitlearn.html) a problemas de *forecasting* es transformar la serie temporal
en un matriz en la que, cada valor, está asociado a la ventana temporal (lags) que le preceden.

<p><img src="./images/transform_timeseries.gif" alt="forecasting-python" title="forecasting-python"></p>

<center><font size="2.5"> <i>Tranformación de una serie temporal en una matriz de 5 lags y un vector con el valor de la serie que sigue a cada fila de la matriz.</i></font></center>
