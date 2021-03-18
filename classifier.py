
#!pip install transformers==3.1.0 --quiet
import transformers
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Classifier:
    """The Classifier"""
    
    def __init__(self):
        '''initialising the class and loading the BERT model from HuggingFace library 
           and giving max embeddings to get for each columns.'''
        # load models
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.bert_model = TFRobertaModel.from_pretrained('roberta-large-mnli')

        # parameters for getting embedding
        self.max_token_dict = {'asp_cat_emb':16,'asp_term_emb':24,'review_emb':50}
        self.src_column_dict = {'asp_cat_emb':'aspect_category','asp_term_emb':'aspect_term','review_emb':'review'}
        
        # loading variable encoder
        self.encoder = LabelEncoder()

        # model
        self.model = self.create_model()


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        # loading the train file
        df = self.loadfile2df(trainfile)
        X = self.get_embeddings_X(df)
        Y = df['sentiment'].values
        Y = self.encoder.fit_transform(np.array(Y).reshape(-1,1))
        
        # optimizer and scheduler
        optim = tf.keras.optimizers.Adam(learning_rate=0.001)
        rlrp = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        
        # loss
        self.model.compile(optimizer=optim, loss= 'sparse_categorical_crossentropy',metrics = ['accuracy'])
        
        # training
        self.model.fit(X, Y,epochs =50,callbacks=[rlrp],verbose = 0)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        # loading the test file
        df2 = self.loadfile2df(datafile)
        X2 = self.get_embeddings_X(df2)
        
        # Predictions
        pred = self.model.predict(X2)
        
        # Encoding to original class labels
        pred = np.argmax(pred,axis= 1)
        y2_pred = self.encoder.inverse_transform(pred)
        
        return list(y2_pred)


    def loadfile2df(self,data_csv):
        '''
        Load the files as pandas dataframe object
        '''
    
        columns = ['sentiment','aspect_category','aspect_term','slice','review']
        df = pd.read_csv(data_csv,sep='\t',names = columns, header = None)
        return df


    def get_embeddings_X(self,df):
        '''
        Load embeddings  from the RoBERTa Model
        '''
    
        emb_list = []
        for col,MAX_LENGTH in self.max_token_dict.items():
            str_inp = df[self.src_column_dict[col]].values
            inputs = self.tokenizer([str(i) for i in str_inp],
                          max_length = MAX_LENGTH,
                          pad_to_max_length = True,return_tensors="pt",truncation=True)
        
            inputs = [np.array(v) for _,v in inputs.items()]
            out = self.bert_model.predict(inputs)
            emb_list.append(out[0])
      
        X = np.concatenate(emb_list,axis =1)
        return X

  
    def create_model(self):
        '''
        Final Classifier NN model. 
        Takes embeddings as input and predicts the class encoded label.
        '''
    

        model= tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2048))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4000))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1250))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(256,activation= tf.nn.leaky_relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64)) #tf.nn.leaky_relu
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(16,activation= tf.nn.leaky_relu))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
        
        return model



