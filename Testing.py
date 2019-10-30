# Uses the saved network information from the Training.py to make predictions and test the wavecasting

import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time

tf.reset_default_graph()

start = time.time()

strt     = 0  # Put the rogue wave in the middle of the predcitions
st       = 0  # To dismiss the first linear part

nblock   = 1
ntest0   = 500
xsize    = 512
nstep    = ntest0 #int(ntest0/nblock)
pred     = nstep
ststep   = 0 
ntest    = ntest0-ststep
ntrain   = 3180+strt-st-nstep

batch_size     = 1

#with open(r'C:\Users\Ehsan\Desktop\Reza\wave1024ss5\ss5_349.csv') as csvfile:
with open(r'C:\Users\Ehsan\Desktop\Reza\wave\ss6_250.csv') as csvfile:

#with open('ss5_010.csv') as csvfile:

    readCSV = csv.reader(csvfile, delimiter=',')
    elev = []
    for row in readCSV:
        elevation = ([float(a) for a in row[0].split()])
#        elevation = ([float(a) for a in row])
        elev.append(elevation)
        
elev      = elev[st:]
maxscalar = np.max(elev)

trainset   =  elev[0:ntrain][:]
testset    =  elev[ntrain:ntrain+ntest*nblock][:]

trainset   =  np.expand_dims(trainset, axis=2)
trainset   =  trainset/maxscalar

tf.reset_default_graph()

X = tf.placeholder(tf.float32,[None,nstep,xsize,1])
y = tf.placeholder(tf.float32,[None,nstep,xsize,1])

kernel_size   = 1
stride        = 1 #kernel_size
out_channels  = xsize
rnn_n_layers  = 1
rnn_type      = 'simple'
bidirectional = False
padding       = 'SAME'
w_std         = 0
scope_name    = 'crnn1'

# Expand to have 4 dimensions if needed
if len(X.shape) == 3:
    X = tf.expand_dims(X, 3)
    
if len(y.shape) == 3:
    y = tf.expand_dims(y, 3)

with tf.variable_scope(scope_name, initializer=tf.truncated_normal_initializer(stddev=w_std)):
     
    n_in_features = xsize
    patches = tf.extract_image_patches(images=X, 
                             ksizes=[1, kernel_size, n_in_features, 1], 
                             strides=[1, stride, n_in_features, 1], 
                             rates=[1, 1, 1, 1], 
                             padding=padding)
    patches = patches[:, :, 0, :]
    patches = tf.expand_dims(patches,axis=3)

    time_steps_after_stride = patches.shape[1].value
    patches = tf.reshape(patches, [batch_size * time_steps_after_stride, kernel_size, n_in_features])
    
    patches = tf.unstack(tf.transpose(patches, [1, 0, 2]))
    
    # Create the RNN Cell
    if rnn_type == 'simple':
      rnn_cell_func = tf.contrib.rnn.BasicRNNCell
    elif rnn_type == 'lstm':
      rnn_cell_func = tf.contrib.rnn.LSTMBlockCell
    elif rnn_type == 'gru':
      rnn_cell_func = tf.contrib.rnn.GRUBlockCell
    if not bidirectional:
      rnn_cell = rnn_cell_func(out_channels)
    else:
      rnn_cell_f = rnn_cell_func(out_channels)
      rnn_cell_b = rnn_cell_func(out_channels)
      
    # Multilayer RNN?
    if rnn_n_layers > 1:
      if not bidirectional:
        rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_n_layers)
      else:
        rnn_cell_f = tf.contrib.rnn.MultiRNNCell([rnn_cell_f] * rnn_n_layers)
        rnn_cell_b = tf.contrib.rnn.MultiRNNCell([rnn_cell_b] * rnn_n_layers)
    
    # The RNN itself
    if not bidirectional:
      outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, patches, dtype=tf.float32)
    else:
      outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(rnn_cell_f, rnn_cell_b, patches, dtype=tf.float32)
    
    if not bidirectional:
      outputs = outputs[-1]
    else:
      half = int(outputs[0].shape.as_list()[-1] / 2)
      outputs = tf.concat([outputs[-1][:,:half], 
                           outputs[0][:,half:]], 
                          axis=1)
    
    # Expand the batch * time-steps back (shape will be [batch_size, time_steps, out_channels]
    if bidirectional:
      out_channels = 2 * out_channels
    outputs = tf.reshape(outputs, [batch_size, time_steps_after_stride, out_channels,1])

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,"512-500steps_ss6-iter10000file700/wave")
   
    trseed = trainset[-nstep-ststep:,:,:]
#    np.savetxt("1024-500steps_ss5/1024-500-feed.csv", trseed[-nstep-ststep:,:,0]*maxscalar, delimiter=",")


    if (ststep == 0):
        X_batch = np.array(trseed[-nstep:,:]).reshape(1, nstep, xsize, 1)
    else:
        X_batch = np.array(trseed[-nstep-ststep:-ststep,:,:]).reshape(1, nstep, xsize, 1)

    y_pred  = sess.run(outputs, feed_dict={X: X_batch})
    
    trseed = np.append(trseed,y_pred[0,ststep:nstep,:,:],axis=0)

    for iteration in range(1,nblock):
        if (ststep == 0):
            X_batch = np.array(trseed[-nstep:]).reshape(1, nstep, xsize, 1)
        else:
            X_batch = np.array(trseed[-nstep-ststep:-ststep,:,:]).reshape(1, nstep, xsize,1)
                       
        y_pred  = sess.run(outputs, feed_dict={X: X_batch})
        trseed  = np.append(trseed,y_pred[0,ststep:nstep,:,:],axis=0)
        
results   = trseed[-np.size(trseed,axis=0)+ststep+nstep:,:,:].reshape(np.size(trseed,axis=0)-ststep-nstep,xsize,1)

results   = results[:,:,0]*maxscalar

x = np.linspace(0,3.1415927535*2,xsize)

for j in range(0,0+1):
    
    i = strt;
    
    if (i == 0 or i == 1):
        plt.plot(x,results[-1:][0],'-b', label='ML')
        plt.plot(x,testset[-1:][0],'--k', label='test')        
    else:
        plt.plot(x,results[-i:-i+1][0],'-b', label='ML')
        plt.plot(x,testset[-i:-i+1][0],'--k', label='test')
    leg = plt.legend();

    L2Norm1 = np.sqrt(np.square(results[-1:][0]-testset[-1:][0]).sum(axis=0))
    L2Norm2 = np.sqrt(np.square(testset[-1:][0]).sum(axis=0))
    print('L2Norm error= ', L2Norm1/L2Norm2)
    
    
    L1Norm1 = abs(results[-1:][0]-testset[-1:][0]).sum(axis=0)
    L1Norm2 = abs(testset[-1:][0]-results[-1:][0]*0).sum(axis=0)
    print('L1Norm error = ', L1Norm1/L1Norm2)

    Hs  = 4*np.std(elev)
    
    print(i,'Hs=',Hs)
    print('Time =    ', time.time()-start)

    elev = np.absolute(elev)


