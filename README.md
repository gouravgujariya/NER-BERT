# NER-BERT
This Notebook is built over the existing dataset present on kaggle under the name of NER_DATASET"https://www.kaggle.com/datasets/namanj27/ner-dataset" and the main object of the Notebook is to showcase the name entitiy recognition feature using bert and hugging face.
 a Named Entity Recognition (NER) model using the Hugging Face Transformers library and the SimpleTransformers wrapper. The code is written in a Jupyter Notebook-like format. To create a readme.md file that explains what the code does, we have used Markdown formatting to describe each step of the process.

 # Named Entity Recognition (NER) using BERT

This repository contains code to train a NER model using the BERT language model from the Hugging Face Transformers library.

## Getting Started

### Prerequisites

- Python (>= 3.6)
- [Transformers library](https://github.com/huggingface/transformers)
- [SimpleTransformers library](https://github.com/ThilinaRajapakse/simpletransformers)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)

### Data

The NER model is trained on the NER dataset located at `/kaggle/input/ner-dataset/ner_datasetreference.csv`.

## Steps

1. Import necessary libraries and set up the device for training (CPU or GPU).

2. Load and preprocess the dataset using pandas. The dataset is read from ner_datasetreference.csv.

3. Prepare the data for training and testing by encoding labels and splitting the data.

4. Configure the NER model using the NERArgs class from SimpleTransformers. Set hyperparameters and initialize the NER model with BERT.

5. Train the NER model using the training data and evaluate it on the test data. Monitor accuracy during training.

6. Evaluate the trained model on the test data and print evaluation results.

7. Make predictions using the trained NER model on sample text.

