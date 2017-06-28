## mytensorflow
A simple examples to learn LSTM in tensor flow.  I generate a sin wave and train the network for 10 periods and try to predict the 11th period.  X is the lagged y by one period.  With these parameters the network trains relatively fast.

### Question 1.  
After training, the prediction of the 11th period is very good.  If I change the starting point of the test_data such that the first point is y=1, the prediction is bad in the beginning but then catches up.  It seems as the state at the beginning of the test is set to zero, which is not compatible with a state where y=1.  The state at the end of training is most likely NOT the state at the beginning of testing after we deploy the model.  How can I initialize the state, and what is that value?  This is similar to training a network for sentence completion, but using it to predict a sentence that already has a couple or words.

### Question 2.  
When I use predict by feeding a dictionary of Xs, am I predicting a set of y of the same length as X or am I predicting $y_{t+1}$ given $X_t$?  The result would be different if every y_t is X at t+1 and I do the prediction recursively.  I want to predict more than one step into the future.

### Question3.
Even if my predictor X is just the lagges 1 period version of y, it seems like the state needs a couple of data points to catch up with the prediction.  Does that mean that my prediction is actually using more than a single point for prediction.  What if I want to use several points in the past to make a prediction, how to I implement it?

### Question 4.  
What if the model has additional feature dependences that are not temporal, for example the online traffic pattern may have different profiles base on the existence of a sale event or not.  How do you model that?

### useful links:
https://stackoverflow.com/questions/43663795/tf-lstm-save-state-from-training-session-for-prediction-session-later
