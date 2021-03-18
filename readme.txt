NLP Exercise 2: Aspect-Based Sentiment Analysis.

Elias Selman
Ignacio Moreno Vergara
Parth Mittal
Sarthak Raisurana.

Our Final approach for this aspect-based sentiment analysis exercise consisted
of using state of the art transformers model as an encoder model for feature
representation and a deep neural network classifier for sentiment analysis
classification.

We experimented with different pretrained transformers from the hugging face
library, most of these models were variations of Bert, GPT2, XLM and RoBERTa.
The best performance was achieved with the pretrained embedding encoder
‘roberta-large-mnli’.

The final classifier was a neural network architecture implemented using the
Keras library. The network consisted of  stacked dense layers alternated with
Leaky Relu (provided better performance with class imbalance) activations and
Dropout layers to avoid overfitting. The optimizer chosen for this task was
Adam because it computes individual adaptive learning rates for different
parameters from estimates of first and second moments of the gradients. Since
we are dealing with mutually exclusive labels (positive-neutral-negative), we
defined a  ‘sparse categorical cross entropy’ loss as our objective function.

To improve the performance, avoid overfitting and adapt the learning rate when
the modes loss begins to converge, we implemented a callback from Tensorflow
(ReduceLROnPlateu), with a monitor of “loss”, a factor of 0.4, patience of 2 and
a min learning rate of 0.0000001. Finally the model achieved the best
performance at 30 epochs.

Finally,  the model performs the best at classifying positive reviews, with a
90% precision. However, the F1 is 82% for negative reviews and 50% for neutral
reviews which are less common in the training set. Overall this generates an
accuracy of 87.1% with the test dataset. 
