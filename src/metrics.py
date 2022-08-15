from fastai.basics import accuracy, BalancedAccuracy, F1Score
METRICS = {'ACC': accuracy, 'BACC': BalancedAccuracy(),
           'F1_weighted': F1Score(average='weighted'), 'F1_macro': F1Score(average='macro')}
