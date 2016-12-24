import tensorflow as tf


input_image = tf.placeholder(tf.float32 , shape = [None , 66 , 200 3])
output_val = tf.placeholder(tf.floate32 , shape = [None, 1])

# Find covolutional layer . 5*5 kernel and 2*2 stides
# First 3 conv layers : has 5 * 5 kernels and 2*2 strides


w_c1 = tf.truncated_normal([5 , 5 , 3 , 24] , stddev = 0.1)
b_c1 = tf.constant(0.1 , [24])

h_c1 = tf.nn.relu(tf.nn.conv2d( input_image , w_c1 , strides = [ 1 , 2 , 2 , 1] , padding = 'VALID') + b_c1)

# 2 conv layer
w_c2 = tf.truncated_normal([5 , 5 , 24 , 36] , stddev = 0.1)
b_c2 = tf.constact(0.1 , [36])

h_c2 = tf.nn.relu(tf.nn.conv2d( w_c1 , w_c2 , strides = [ 1 , 2 , 2 , 1] , padding = 'VALID') + b_c2)

# 3 conv layer
w_c3 = tf.truncated_normal([5 , 5 , 36 , 48] , stddev = 0.1) 
b_c3 = tf.constant(0.1 , [48])

h_c3 = tf.nn.rele(tf.nn.conv2d( w_c2 , w_c3  , strides = [1 , 2 , 2 , 1]  , padding = 'VALID') + bc3)

#The last 2 convoluational layers : has 3 * 3 kernels and 1 * 1 strides

# 4th conv layer
w_c4 = tf.truncated_normal([3 , 3 , 48 , 64] , stddev = 0.1)
b_c4 = tf.constanct(0.1 , [64])

h_c3 = tf.nn.relu(tf.nn.conv2d( w_c3 , w_c4 , strides = [1 , 1 , 1 , 1] , padding = 'VALID') + bc_4)

# 5 conv layer
w_c5 = tf.truncated_normal([3 , 3 , 64 , 64] , stddev = 0.1)
b_c5 = tf.constanct(0.1 , [64])

h_c5 = tf.nn.relu(tf.nn.conv2d( w_c4 , w_c5 , strides = [1 , 1 , 1 ,1] , padding = 'VALID') + bc_5)

# Fully connected layer 

#FC1 
w_fc1 = tf.truncated_normal([1152, 1164] , stddev = 0.1)
b_fc1 = tf.constant(0.1 , [1164])

# Flatten the output of the last convolutional layer and input it into the fully connected layer.;

h_c5_flat = tf.reshape(h_c5 ,[ -1 , 1152])
h_fc1 = tf.nn.relu(tf.matmul(h_c5_flat , w_fc1) + b_fc1)



