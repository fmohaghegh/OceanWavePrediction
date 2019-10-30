import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time

def next_batch(training_data,batch_size,steps,pred,nfiles):
    filelength = int(len(training_data)/nfiles);
    randfile   = np.random.randint(nfiles);
    rand_start = np.random.randint(0,filelength-pred-steps+1);
    rand_start = rand_start+randfile*filelength;
    batch = np.array(training_data[rand_start:rand_start+pred+steps]).reshape(1,pred+steps,np.shape(training_data)[1],1)
        
    for i in range(1,batch_size):
        randfile   = np.random.randint(nfiles);
        rand_start = np.random.randint(0,filelength-pred-steps+1);
        rand_start = rand_start+randfile*filelength;
        batch = np.append(batch,np.array(training_data[rand_start:rand_start+pred+steps]).reshape(1,pred+steps,np.shape(training_data)[1],1),axis=0)       
    return batch[:, :-pred], batch[:, pred:]

tf.reset_default_graph()

start = time.time()

nfiles   = 250
num_iter = 10000
nblock   = 1
ntest0   = 500
xsize    = 512
nstep    = int(ntest0/nblock)
pred     = nstep
ststep   = 0 
ntest    = ntest0-ststep
ntotal   = 6400
st       = 500
ntrain   = nfiles*(ntotal-st)-500


num_inputs     = 1
num_outputs    = 1
learning_rate  = 0.00001
batch_size     = 10

elev           = []

for i in range(1,nfiles+1):
    print(i)
    iname = np.random.randint(1,nfiles+1)
    name  = r'C:\Users\Ehsan\Desktop\Reza\wave\ss6_{0:01d}'.format(iname+55)
    name  = name + '.csv'
    count = 0 
    
    with open(name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            elevation = ([float(a) for a in row[0].split()])
            count = count+1
            if (count > 500):
                elev.append(elevation)
            
maxscalar = np.max(elev)

trainset   =  elev[0:ntrain][:]
testset    =  elev[ntrain:ntrain+ntest][:]

trainset   =  np.expand_dims(trainset, axis=2)
trainset   =  trainset/maxscalar

tf.reset_default_graph()

X = tf.placeholder(tf.float32,[None,nstep,xsize,1])
y = tf.placeholder(tf.float32,[None,nstep,xsize,1])

kernel_size   = 1
stride        = 1#kernel_size
out_channels  = xsize
rnn_n_layers  = 1
rnn_type      = 'simple'
bidirectional = False
padding       = 'VALID'
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

##      patches = tf.expand_dims(patches, 3)
#      cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=ntest, activation=tf.nn.relu), output_size=out_channels)
#      outputs,states = tf.nn.dynamic_rnn(cell,patches,dtype=tf.float32)

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


               

loss      = tf.reduce_mean(tf.square(outputs-y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train     = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    loss_history = []
    loss_history1 = []
 
    for iteration in range(num_iter):  
        X_batch, y_batch = next_batch(trainset,batch_size,nstep,pred,nfiles)

        _,los = sess.run([train,loss], feed_dict={X: X_batch, y: y_batch})
        loss_history.append(los)

        if iteration % 250 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
            loss_history1.append(los)
        
        if (mse<0.0001):
            break;
        

    saver.save(sess, "512-500steps_ss6-iter10000file700/wave")
   
    trseed = trainset[-nstep-ststep:,:,:]
    
    print('Time =    ', time.time()-start)
