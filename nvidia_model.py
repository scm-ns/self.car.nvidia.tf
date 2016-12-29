import tensorflow as tf


"""
    Creates an exact replica of the nvidia paper
"""
class nvidia_model():
    def __init__(self ):

        self.input_image = tf.placeholder(tf.float32 , shape = [None , 66 , 200 ,3])
        self.output = tf.placeholder(tf.floate32 , shape = [None, 1])

        # Find covolutional layer . 5*5 kernel and 2*2 stides
        # First 3 conv layers : has 5 * 5 kernels and 2*2 strides

        self.w_c1 = tf.truncated_normal([5 , 5 , 3 , 24] , stddev = 0.1)
        self.b_c1 = tf.constant(0.1 , [24])
        self.h_c1 = tf.nn.relu(tf.nn.conv2d( self.input_image , self.w_c1 , strides = [ 1 , 2 , 2 , 1] , padding = 'VALID') + self.b_c1)

        # 2 conv layer
        self.w_c2 = tf.truncated_normal([5 , 5 , 24 , 36] , stddev = 0.1)
        self.b_c2 = tf.constact(0.1 , [36])
        self.h_c2 = tf.nn.relu(tf.nn.conv2d( self.w_c1 , self.w_c2 , strides = [ 1 , 2 , 2 , 1] , padding = 'VALID') + self.b_c2)

        # 3 conv layer
        self.w_c3 = tf.truncated_normal([5 , 5 , 36 , 48] , stddev = 0.1) 
        self.b_c4 = tf.constant(0.1 , [48])
        self.h_c3 = tf.nn.rele(tf.nn.conv2d( self.w_c2 , self.w_c3  , strides = [1 , 2 , 2 , 1]  , padding = 'VALID') + self.bc3)


        #The last 2 convoluational layers : has 3 * 3 kernels and 1 * 1 strides

        # 4th conv layer
        self.w_c4 = tf.truncated_normal([3 , 3 , 48 , 64] , stddev = 0.1)
        self.b_c4 = tf.constanct(0.1 , [64])
        self.h_c3 = tf.nn.relu(tf.nn.conv2d( self.w_c3 , self.w_c4 , strides = [1 , 1 , 1 , 1] , padding = 'VALID') + self.bc_4)

        # 5 conv layer
        self.w_c5 = tf.truncated_normal([3 , 3 , 64 , 64] , stddev = 0.1)
        self.b_c5 = tf.constanct(0.1 , [64])
        self.h_c5 = tf.nn.relu(tf.nn.conv2d( self.w_c4 , self.w_c5 , strides = [1 , 1 , 1 ,1] , padding = 'VALID') + self.bc_5)

        # COntrol the dropout for the next few fully connected layers
        self.keep_prob = tf.placeholder(tf.float32)


        # Flatten the output of the last convolutional layer and input it into the fully connected layer.;
        self.h_c5_flat = tf.reshape(self.h_c5 ,[ -1 , 1152])

        # Fully connected layer 

        #FC1 
        self.w_fc1 = tf.truncated_normal([1152, 1164] , stddev = 0.1)
        self.b_fc1 = tf.constant(0.1 , [1164])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_c5_flat , self.w_fc1) + self.b_fc1)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1 , keep_prob)

        # FC2
        self.w_fc2 = tf.truncated_normal([1164,100] , stddev = 0.1)
        self.b_fc2 = tf.constant([100])
        self.h_fc2 = tf.nn.relu(tf.malmul(h_fc1_drop , w_fc2) + self.b_fc2)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2 , keep_prob)

        #FC 3
        self.w_fc3 = tf.truncated_normal([100,50] , stddev = 0.1)
        self.b_fc3 = tf.constant([50])        
        self.h_fc3 = tf.nn.relu(tf.malmul(h_fc2_drop , w_fc3) + self.b_fc3)
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3 , keep_prob)

        #FC 4
        self.w_fc4 = tf.truncated_normal([50,10] , stddev = 0.1)
        self.b_fc4 = tf.constant([10])        
        self.h_fc4 = tf.nn.relu(tf.malmul(h_fc3_drop , w_fc4) + self.b_fc4)
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4 , keep_prob)


        self.w_fc5 = tf.truncated_normal([10,1] , stddev = 0.1)
        self.b_fc5 = tf.constant([1])
        
        self.output_pred = tf.atan(tf.matmul(self.h_fc4_drop , self.w_fc5) + self.b_fc5)
        self.output_pred = tf.nn.softmax(self.output_pred); 


        self.train_coeff = tf.trainable_variables();
        self.l2_norm_const = 0.001;

        self.loss = tf.sum( tf.reduce_mean(tf.square(tf.sub(self.output , self.output_pred))) , 
                    tf.add_n([tf.nn.l2_loss(t) for t in self.train_coeff])* self.l2_norm_const )

        self.cross_entropy = - tf.reduce_sum(self.output * tf.log(self.output_pred))
        self.correct_pred = tf.equal(tf.argmax(self.output_pred , 1)  , tf.argmax(self.output , 1))
        self.accuracy = tf.reduce_mean(tf.correct_pred)  
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)






