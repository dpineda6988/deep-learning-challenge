# Deep-Learning-Challenge
# Credit-Risk-Classification

The creation and evaluation of several deep learning neural networks that were built using Tensorflow to serve as a a binary classifier that can predict whether applicants for business venture funding will be successful.

## Overview
The fictional nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 

Alphabet Soup’s business team, has provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

Using this dataset, several deep learning neural network models were created in an effort to create a binary classifier that could be used as a tool to predict the success of applicants who could potentially be funded by Alphabet Soup with an accuracy of greater than 75%.

## Results

Data Preprocessing
* What variable(s) are the target(s) for your model?    

Since the purpose of the model was to predict the success of funded applicants, the field 'IS_SUCCESSFUL' was the target variable for the model.

* What variable(s) are the features for your model?

The features of the model were composed of almost all other fields within the dataset.  These variables were as follows:
   * APPLICATION_TYPE—Alphabet Soup application type
   * AFFILIATION—Affiliated sector of industry
   * CLASSIFICATION—Government organization classification
   * USE_CASE—Use case for funding
   * ORGANIZATION—Organization type
   * STATUS—Active status
   * INCOME_AMT—Income classification
   * SPECIAL_CONSIDERATIONS—Special considerations for application
   * ASK_AMT—Funding amount requested
    
Each of these variables potentially had a degree of correlation with the sucess or failure of a funded applicants business venture and were all kept within the model.  However, it is important to note that within the APPLICATION field there were several application types that had a low number of relative occurrences:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>APPLICATION_TYPE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T3</th>
      <td>27037</td>
    </tr>
    <tr>
      <th>T4</th>
      <td>1542</td>
    </tr>
    <tr>
      <th>T6</th>
      <td>1216</td>
    </tr>
    <tr>
      <th>T5</th>
      <td>1173</td>
    </tr>
    <tr>
      <th>T19</th>
      <td>1065</td>
    </tr>
    <tr>
      <th>T8</th>
      <td>737</td>
    </tr>
    <tr>
      <th>T7</th>
      <td>725</td>
    </tr>
    <tr>
      <th>T10</th>
      <td>528</td>
    </tr>
    <tr>
      <th>T9</th>
      <td>156</td>
    </tr>
    <tr>
      <th>T13</th>
      <td>66</td>
    </tr>
    <tr>
      <th>T12</th>
      <td>27</td>
    </tr>
    <tr>
      <th>T2</th>
      <td>16</td>
    </tr>
    <tr>
      <th>T25</th>
      <td>3</td>
    </tr>
    <tr>
      <th>T14</th>
      <td>3</td>
    </tr>
    <tr>
      <th>T29</th>
      <td>2</td>
    </tr>
    <tr>
      <th>T15</th>
      <td>2</td>
    </tr>
    <tr>
      <th>T17</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

A similar case also occurred in the CLASSIFICATION field, where there were a large number of classification types had very few occurrences (as low as 1) in the dataset.  To mitigate the potential influence of these rare occurences being outliers and skewing the results, these were consolidated into an 'OTHER' bin within each respective field.

* What variable(s) should be removed from the input data because they are neither targets nor features?

Besides the consolidation of certain cases within CLASSIFICATION and APPLICATION, the fields for EIN and NAME were both excluded entirely from the data.  Each case for these fields were compeltely unique/distinct and, in a practical sense, have no correlation or impact upon the success of a funded venture.

Compiling, Training, and Evaluating the Model
* How many neurons, layers, and activation functions did you select for your neural network model, and why?

Our initial model consisted of three layers total with the first layer containing 80 neurons, the second layer containing 30 neurons, and the last/ouput later containing one neuron.  The three layer structure and the large number of neurons was used in order to include additional levels of processing given the complexity and breadth of the features involved (after processing and encoding, the features read into the model came to a total of 49).  As for the activation functions, the first layer utilized the ReLu function, due to its relatively fast processing, application for classifications, and the lack of negative numbers in the training data.  The remaining layers used a sigmoid activation function due to its more robust application for binary classifications which was the aim of the model.


* Were you able to achieve the target model performance?

The initial trained model, when evaluated using the test data, yielded an accuracy of ~72.87%.  This failed to meet the 75% threshold and required additional optimizations.

* What steps did you take in your attempts to increase model performance?



A total of three attempts were made to optimize the model and reach the 75% threshold.  The first attempt required revisiting the dataset and rethinking how it was being organized.  While the some of the data in the CLASSIFICATION and APPLICATION_TYPE fields was consolidated into OTHER bins, the rare occurrences of certain cases may have skewed the OTHER bins in a way that negatively impact accuracy.  By lowering the cutoff frequency for grouping data into an OTHER bin, the model was able to improve slightly to ~72.94 accuracy based on the test data.

The second attempt build upon the previous one, this time by increasing the number of training epochs.  While it is may still be possible to further make adjustments to the training data, increasing the number of training epochs could elicit better performance if the model's three layers were not optimally trained.  Increasing the training epochs from 100 to 400 resulted in the model improving further to ~72.99%.

The third attempt built on the previous work even further by an additional layers and increasing the number of neurons (first layer: 100 neurons, second layer: 60 neurons, third layer: 30 neurons, output layer: 1 neuron).  All layers still used a sigmoid activation function with the exception of the first layer which still used a ReLu function.  The additional layers and neurons actually resulted in a decreased accuracy rate of ~72.72% based on the test data.  This could be the result of not increasing the training epochs further to account for the additional layers or it may be that the model could be overfitting to the training data.

![alt text](<Bar Chart_Accuracy by Model Version.png>)

## Summary
Overall, our attempts at creating a binary classfier using a deep learning neural network model fell short of the desired 75% accuracy rate.  Despite attempts at optimizing the data through reorganizing bins of rare occurences and adjusting the hyperparamters of the model via layers, neurons, and training epochs, any improvements to the model's accuruacy were marginal at best.  While it is possible that the data could be further manipulated to account for any additional outliers or confusing variables that could be disrupting the model, it may be worthwhile to explore using another model.  A Random Forest model could potentially be an effective method of creating the desire binary classifier.  Random forest models are an effective model for classification and come with additional strengths that a deep learning neural net lacks.  Given that Random Forests are composed of numerous weak classifiers, each weak classifier is trained on different pieces of data so the overall model is robust against overfitting.  This structure also makes them robust to outliers and non-linear data that could potentially through off the results of a neural network.  Additionally, the model is able to run efficeintly on large databases so the load of data utilized here can easily be handled and processed.


## Dependencies
* Python (w/ jupyter notebook) and the following libraries/framworks
    * keras_tuner
    * pandas
    * tensorflow

## Installing & Execution
The repository files can be downloaded to open and execute the 'AlphabetSoupCharity.ipynb' and 'AlphabetSoupCharity_Optimization.ipynb' files.

## Authors

Daniel Pineda

## Acknowledgments
Deep-Learning-Challenge was created as an assignment for the University of California, Irvine Data Analytics Bootcamp - June 2024 Cohort under the instruction and guidance of Melissa Engle (Instructor) and Mitchell Stone (TA).
The practical exercises and coding examples demonstrated through the bootcamp helped inform and inspire the code for this project.