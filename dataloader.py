import numpy as np
import os
import pickle

def load_data(path, mode='train'):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1,6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data']
            label = images_dict[b'labels']
            labels.extend(label)
            images.extend(data)
        
        return np.array(images), np.array(labels)
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        
        return np.array(test_data), np.array(test_labels)
    else:
        raise NotImplementedError(f"Provide a valid mode for load data (train/test)")
        
def one_hot_encoding(labels):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    num_classes = len(np.unique(labels))
    
    
    y_hot = np.zeros((len(labels),num_classes))
    y_hot[np.arange(len(labels)), labels] = 1
    
    return y_hot
class DataLoader:
    ''' Load data, make validation, one hot, and normalize'''
    def __init__(self):
        self.X_train, self.y_train = load_data('data/')
        self.X_test, self.y_test = load_data('data/', mode = 'test')
        
        self.make_validation()
        self.one_hot_encoding()
        
        self.normalize_data()
    def split_to_k(self, k = 10):
        ''' split to k equal portions'''
        k_set_index = [[] for _ in range(k)]
        
        for i,category in enumerate(np.unique(self.y_train)):
            indices = np.where(self.y_train == category)[0]
            
            # shuffle
            np.random.shuffle(indices)
            #print(indices)
            
            
            # split into k sets
            k_sets = np.array_split(indices, k) 
            
            #print(len(k_sets), len(k_set_index), len(k_sets[0]), len(indices))
            
            for new_index, fold in zip(k_sets, k_set_index):
                fold.append(new_index)
            #print(len(fold))
                    
            
        # a list of np.arrays
        k_set_index=[np.concatenate(index) for index in k_set_index]
        
        #print(len(k_set_index))
        return k_set_index
    
    def one_hot_encoding(self):
        self.y_train = one_hot_encoding(self.y_train)
        self.y_test = one_hot_encoding(self.y_test)
        self.y_val =one_hot_encoding(self.y_val)
        
        
    def make_validation(self, fraction=10):
        k_set_index = self.split_to_k(k = fraction)
        
        val_index = k_set_index[0]
        
        train_index = np.concatenate(k_set_index[1:])
        
        self.X_val = self.X_train[val_index, :]
        self.y_val = self.y_train[val_index]
        self.X_train = self.X_train[train_index,:]
        self.y_train = self.y_train[train_index]
    
    def normalize_data(self):
        self.mean = np.mean(self.X_train, axis = 0)
        self.std = np.std(self.X_train, axis = 0)
        
        # apply on all three
        self.X_train = (self.X_train-self.mean)/self.std
        self.X_test = (self.X_test-self.mean)/self.std
        self.X_val = (self.X_val-self.mean)/self.std
    
        