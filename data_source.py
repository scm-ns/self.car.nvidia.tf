# Provides the data to the training set
import pandas as pd
import cv2


class data_source():
    def __init__():
        self.xs = []
        self.ys = []

        # To avoid repetitions of images in different batches
        self._train_batch_ptr = 0 ; 
        self._val_batch_ptr = 0 ; 

        # To load data from udacity 
        self.data_set_dir = "./data_set";
        self.df = pd.read_csv("./data_set/data.csv")
        self.df = self.df[df.frame_id == 'center_camera']
        
        self.xs = [self.data_set_dir + idx for idx in self.df.filename]
        self.ys = df.angle

        self.num_images = len(xs)
        
        self._data_set = list(zip(self.xs , self.ys))
        random.shuffle(self._data_set)
        self.xs , self.ys = zip(*_data_set) 

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(ys) * 0.8)]

        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(ys) * 0.2):]

        self.num_train_img = len(train_xs)
        self.num_val_img = len(val_xs)

    
    def load_train_batch(batch_size):
        x_out = []
        y_out = []

        for i in range(0 , batch_size):
            img = cv2.imread(self.train_xs[(self._train_batch_ptr + i ) % self.num_train_img])
            distorted = img 
            x_out.append(cv2.resize(distorted , (200 , 66))/255.0)
            y_out.append([self.train_ys[(self._train_batch_ptr + i) % self.num_train_img]])
        self._train_batch_ptr += batch_size ; 
        return x_out , y_out;

           
    def load_val_batch(batch_size):
        x_out = []
        y_out = []
        for i in range(0 , batch_size):
            x_out.append(cv2.resize(cv2.imread(val_xs[(self._val_batch_ptr + i) % self.num_val_img]) , (200,60)) / 255.0)
            y_out.append([self.val_ys[(self._val_batch_ptr + i) % num_val_img]])
        self._val_batch_ptr += batch_size ; 
        return x_out , y_out 
            
    # Temporary solution. See how the test set for udacity is set up
    def load_test_set():
        return load_train_batch(100)

