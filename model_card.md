
## Model Card
This card was generated on _2022-05-15 14:31_     to summarize last model performance for the project __Census Income Data Set__

## Model Details

The predictive model is based on three part :

    - A label binarizer to convert the target income column into a binary data
    - An encoder to convert categorical features into digital ones
    - A random forest classifier to predict the income category ( > or < 50k)
    

## Intended Use

Predict whether income exceeds $50K/yr based on census data.


## Training Data

The used dataset and relative details can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)
    

## Evaluation Data

The dataset was split randomly into 80% data for training and 20% for model evaluation.
    

## Metrics
The current metrics of the model is as following:

    - Precision : 0.780
    - Recall    : 0.634
    - F1 Score  : 0.700
    

## Ethical Considerations

    Even though this data is completely anonymous and open to the public,
    it contains sensitive data relating to people's private lives such
    as their income levels, origins...

    Users of this data should pay particular attention to the biases
    that may exist or be introduced in order not to favor or disadvantage
    one category of the population over another.
    

## Caveats and Recommendations

    The use and application of all or part of this model and these results
    are the entire responsibility of the user.

    