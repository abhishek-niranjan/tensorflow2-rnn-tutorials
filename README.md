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
   * ``` utils/attention.py``` contains ```BahdanauAttention``` and ```LuongAttention``` class. These attention mechanisms have also been explained in [encoder-decoder with attention notebooks](lesson 4 and lesson 5)
   
2. ```tutorial-notebooks``` directory contains all the jupyter notebooks.
   * [Notebook-1 (text-processing):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/1_text_processing.ipynb) explains how to   use ```tf.keras.preprocessing``` module to preprocess textual corpus and prepare ```tf.data.Dataset``` objects.
   * [Notebook-2 (embedding-and-classification):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/2_embeddings_and_classification.ipynb) Fundamentals of a many-to-one recurrent neural network and how to program it in tensorflow-2.0 
   * [Notebook-3 (encoder-decoder):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/3_basic_encoder_decoder.ipynb) We build a encoder-decoder architecture with ```tf.keras.layers.GRU``` as the base recurrent layer in both encoder and decoder.
   * [Notebook-4 (Bahdanau Attention):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/4_enc_dec_with_BahdanauAttention.ipynb) We describe the Bahdanau Attention mechanism and how it is incorporated in an encoder-decoder architecture. 
   * [Notebook-5 (Luong Attention):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/5_enc_dec_with_LuongAttention.ipynb) We describe Luong Attention and how is it incorporated in an encoder-decoder systm. 
   * [Notebook-6 (Stacked Encoder-Decoder):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/6_stacked_biRNN.ipynb) We build stacked(more than one layer) encoder with Bi-directional LSTM layers and stacked decoder with Unidirectional LSTM layers. 
   * [Notebook-7 (GoogleNMT):](https://github.com/abhishek-niranjan/tf2-rnn-tutorials-for-beginners/blob/master/tutorial-notebooks/7_GoogleNMT_architecture.ipynb) We program the architecture described in [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf). The encoder and decoder stacks contain a total of 8 layers and residual connections from *3<sup>rd</sup>* layer onwards.

