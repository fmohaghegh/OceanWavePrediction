# Ocean Wave Prediction
This repository uses recurrent neural nets to predict the ocean waves from the previous data. It takes the spatiotemporal data from the simulation, and applies Recurrent Neural Net in the time series data. Each node in the time seris data is the spatial domain.


This code is the revised version of CRNN in the paper:
Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data (https://arxiv.org/abs/1602.05875)

Calling the below function is equivalnet to applying one CRNN layer. For a deep model with a few
CRNN layers, the function should be invoked multiple times. 
Given a tensor, the function extracts patches of `kernel_size` time-steps, and processed each 
with one or more recurrent layers. The hidden state of the recurrent neural network is then 
returned as the feature vector representing the path. 

Args:

  tensor: The tensor to perform the operation on, shape `[batch, time-steps, features]`
          or `[batch, time-steps, features, 1]`.
          
  kernel_size: The number of time-steps to include in every patch/window (same as in standard 1-D convolution).
  
  stride: the number of time-steps between two consecutive patches/windows (same as in standard 1-D convolution).
  
  out_channels: The number of extracted features from each patch/window (in standard 1-D convolution 
                known as the number of feature maps), which is the hidden dimension of the recurrent 
                layers that processes each patch/window.
                
  rnn_n_layers: The number of recurrent layers to process the patches/windows. 
    (in the original paper was always =1). 
    
  rnn_type: Type of recurrent layers to use: `simple`/`lstm`/`gru`
  
  bidirectional: Whether to use a bidirectional recurrent layers (such as BLSTM, when the rnn_type is 'lstm'). 
                 If True, The actual number of extracted features from each patch/window is `2 * out_channels`.
                 
  w_std: Weights in the recurrent layers will be initialized randomly using a Gaussaian distribution with
         zero mean and a standard deviation of `w_std`. Biases are initialized with zero. 
         
  padding: `SAME` or `VALID` (same as in standard 1-D convolution).
  
  scope_name: For variable naming, the name prefix for variables names.  
  
Returns:
A 3-D `Tensor` with shape `[batch, time-steps, features]`, similarly to the output of a standard 1-D convolution. 

Only Training.py and Testing.py are for the Machine Learning Part. Other files are MATLAB files to preform the simulation. wave_simulation_hos.f is a g95 Fortran file required to run Initial_spectrum.m which is the main file for the simulation. 

