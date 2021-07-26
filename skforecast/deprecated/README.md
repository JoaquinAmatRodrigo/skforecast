
In the deprecated version of `ForecasterAutoregMultiOutput`, the data matrices used to train each model share the information of exogenous variables for all steps. Althougt this is not including future information (exogenous variables are known in advance), the number of predictors increase rapidly.


<p><img src="../../images/diagram_skforecast_multioutput_deprecated.jpg"></p>

<center><font size="2.5"> <i>Deprecated strategy of transformation needed to train a direct multi-step forecaster.</i></font></center>
<br><br>

<br><br>

Istead, the current strategy include for each model only the exogenous values associated with the current step.

<p><img src="../../images/diagram_skforecast_multioutput.jpg"></p>

<center><font size="2.5"> <i>Current strategy of transformation needed to train a direct multi-step forecaster.</i></font></center>
<br><br>

<br><br>
