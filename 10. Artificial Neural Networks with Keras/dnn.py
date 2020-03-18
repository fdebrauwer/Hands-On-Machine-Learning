from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from ocs import OneCycleScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import matplotlib.pyplot as plt

class SequentialNetwork:
    
    
    def __init__(self, dropout=0.2, hidden_layers=3):
        self.dropout = dropout
        self.hidden_layers = range(hidden_layers)
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        for layers in self.hidden_layers:
            self.model.add(Dropout(rate=self.dropout))
            self.model.add(Dense(100, activation='elu', kernel_initializer='he_normal')) 
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    
    def train(self, train_images, train_labels, test_images, test_labels, n_epochs=100, 
              batch_size=64, validation_split=0.1, one_cycle=True, early_stopping=True):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        if one_cycle == True :
            onecycle = OneCycleScheduler(len(train_images) // self.batch_size * self.n_epochs, max_rate=0.05)
        
        if early_stopping == True :
            earlystopping = EarlyStopping(patience=10, restore_best_weights=True)

        start_time = time.time()
        
        history = self.model.fit(
            train_images, 
            train_labels, 
            epochs=self.n_epochs, 
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[onecycle, earlystopping]
        )

        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=0)

        print('\nTest accuracy:', test_acc)
        print(self.model.summary())
        print('\nIt took :', round(time.time()-start_time,3), 'to train...')
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
