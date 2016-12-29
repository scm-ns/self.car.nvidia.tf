# Provides the data to the training set




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
        self.df = self.df[df.frame_id = 'center_camera']
        
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
        

