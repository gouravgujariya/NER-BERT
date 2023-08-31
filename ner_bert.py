# -*- coding: utf-8 -*-
"""ner-bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FbWYYWe37LbY-eXdsXfRULhJymGhf9gZ
"""

!pip install transformers seqeval[gpu]

!pip install simpletransformers

from sklearn.metrics import accuracy_score
import torch

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

data = pd.read_csv('/kaggle/input/ner-dataset/ner_datasetreference.csv',encoding = 'unicode_escape')
data.head()

data.count()

print('number of tags: {}'.format(len(data.Tag.unique())))
freq = data.Tag.value_counts()
freq

data = data.fillna(method = 'ffill')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data['Sentence #'] = LabelEncoder().fit_transform(data['Sentence #'])
data.head()

data.rename(columns = {'Sentence #':'sentence_id',
                       'Word':'words',
                       'Tag':'label'},inplace = True)

data

data['label'] = data['label'].str.upper()

X = data[['sentence_id','words']]
y = data['label']

X_train,X_test,y_train,y_test = train_test_split(X,y)

train_data = pd.DataFrame({"sentence_id":X_train["sentence_id"],"words":X_train["words"],"labels":y_train})
test_data = pd.DataFrame({"sentence_id":X_test["sentence_id"],"words":X_test["words"],"labels":y_test})

len(train_data)

from simpletransformers.ner import NERModel , NERArgs

labels = data['label'].unique().tolist()
labels

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 32
args.eval_batch_size = 32

model = NERModel('bert','bert-base-cased',labels = labels , args = args)

model.train_model(train_data , eval_data = test_data , acc = accuracy_score)

result, model_outputs, preds_list = model.eval_model(test_data)

result

prediction, model_output = model.predict(["Alexanderia is the beauty of north africa"])
prediction