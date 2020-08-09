The tutorial notebooks in this repository are aimed to help beginners program basic recurrent neural networks(RNNs) for textual problems in tensorflow-2. 

## Prerequisite 
  * Understanding of basic textual processing methods, i.e familiarity with tokenization, etc.
  * Functioning of basic RNN, GRU, LSTM cells
  * Fundamentals of RNN based Encoder-Decoder Architectures 
  * Attention mechanism, mainly [Bahdanau Attention](https://arxiv.org/abs/1409.0473) and [Luong Attention](https://arxiv.org/abs/1508.04025)
  * A basic idea of beam search algorithm.
  
## Contents

1. ```utils``` directory contains helper classes and functions.  
   * ```utils/dataset.py``` contains ```NMTDataset``` class which creates  training and validation ```tf.data.Dataset``` splits and also returns input and target side tokenizers (```tf.keras.preprocessing.text.Tokenizer```). The working of ```utils/dataset.py``` have been explained in first notebook on [text-processing notebook](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/1_text_processing.ipynb)
   * ``` utils/attention.py``` contains ```BahdanauAttention``` and ```LuongAttention``` class. These attention mechanisms have also been explained in [encoder-decoder with attention notebook](lesson 4)
   
2. ```tutorial-notebooks``` directory contains all the jupyter notebooks.
   * [Notebook-1 (text-processing):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/1_text_processing.ipynb) explains how to   use ```tf.keras.preprocessing``` module to preprocess textual corpus and prepare ```tf.data.Dataset``` objects.
