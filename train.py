import tensorflow as tf
import nvidia_model 
import data_source

def train_net(max_iter = 20 , batch_size = 50):
        model = nvidia_model()
        ds = data_source()
        n_ter = 0 ;
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver();

            while(n_ter < max_iter):
                for batch_idx in range(ds.total_samples / batch_size):
                    xs , ys = ds.load_train_batch(batch_size)

                    # Train the model using hte optimizer setup in the nvidia_model.py file
                    sess.run(nvidia_model.optimizer , feed_dict = {model.input : xs , model.output : ys , model.keep_prob : 0.8})

                    if batch_idx % 10 == 0:
                        train_acc = model.accuracy.eval(session = sess , feed_dict = {model.input : xs , model.output : ys , model.keep_prob : 1 })
                        print("step : %d training acc : %g" %(batch_idx , train_acc))

                    if batch_idx % 15 == 0:
                        xs_val , ys_val = ds.load_val_batch(batch_size)
                        train_acc = model.accuracy.eval(session = sess , feed_dict = {model.input : xs_val , model.output : ys_val , model.keep_prob : 1 })
                        print("step : %d validation acc : %g" %(batch_idx , train_acc))

                     
                # Write model at end of each iteration 
                if not os.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                checkpoint_filename = "model" + n_iter + ".ckpt"
                checkpoint_path = os.path.join(LOGDIR , checkpoint_filename)
                filename = saver.save(sess , checkpoint_path)
                print("model saved at end of iter %d : @ : %g" %(n_iter , filename)); 

                n_ter += 1; 

            # Test the trained model, at the end of all iterations
            xs_test , ys_test = ds.load_test_batch(); 
            test_acc = model.accuracy.eval(session = sess , feed_dict = {model.input : xs_test , model.output : ys_test , model.keep_prob : 1.0}) 
