import numpy as np
try:
    import tensorflow.keras as ke
except:
    import keras as ke
    print()
    print('\u001b[31m' + "현재 버전은 keras가 작동되지 않습니다.\n텐서플로우 라이브러리를 업데이트를 통해 버전을 올리세요!" + '\u001b[0m')
    print()

import tensorflow as tf
import tensorflow.keras as ke
ke.Sequential.compile

class test:
    def __init__(self):
        print("__testing__")

    def set_Data(self,title,pria):
        self.title = title
        self.pria = pria

    def printing(self):
        print(self.title)
        print(self.pria)

'''a = test()
a.set_Data("modelsmodels","modelsa")
a.printing()'''


class auto_tf_ANN:

    def __init__(self):
        print('\u001b[32m'+"__edit_tensoeflow__"+'\u001b[0m')
        pass

    def version_set():
        print("라이브러리 제작시 사용된 텐서플로우 버전 : 2.7.0")
        seli = tf.__version__
        print("현재 텐서플로우 버전 : ", seli)
        if str(seli) < "2.7.0":
            print()
            print('\u001b[31m' + "하위 버전의 경우 keras가 작동되지 않을수 있습니다.\n텐서플로우 라이브러리를 업데이트하여 버전을 올리세요!" + '\u001b[0m')
            print()

    def keras_Sequential():
        return ke.Sequential()

    def auto_acc(self, x, y, acc= 1.00,  save = False, save_name = "auto_Model.h5"):

        models = self.models

        total = 0
        print(acc)
        while True:
            l = models.fit(x,y,epochs=1)
            lossoff = l.history['accuracy']
            if acc == float(max(lossoff)):
                print(min(lossoff))
                break

        if save:
            models.save(save_name)

        self.models = models

        return models

    def auto_loss(self, x,y, loss = 0.01, save = False, save_name = "auto_Model.h5", callbacks = False):

        models = self.models

        print("loss : ", loss)
        #print(models)
        while True:

            if callbacks:
                ack = self.call
                l = models.fit(x,y,epochs=5,callbacks=[ack])
            else:
                l = models.fit(x,y,epochs=5)

            lossoff = l.history['loss']

            if loss > float(max(lossoff)):
                print(min(lossoff))
                break
        if save:
            models.save(save_name)

        self.models = models

        return models

    def auto_label(self, date_Frame,named):

        l = ''
        p = ''
        labels = []

        Space = 0

        ic = 0

        for i in range(len(date_Frame)):
            ic = i
            p = date_Frame[named][ic]


            if p != l:
                print(p)
                labels.append(p)
                Space += 1
            l = p
            date_Frame[named][ic] = Space - 1


        #print(Space)
        self.mond = labels
        return  date_Frame, labels


    def callbak(self,x,y):
        clobac = ke.callbacks.EarlyStopping(monitor='val_loss',patience=3)
        self.call = clobac
        models = self.models

        while True:

            l = models.fit(x,y,epochs=1000,callbacks=[clobac])
            lossoff = l.history['loss']
            if lossoff[999] < 0.01:
                break
        self.models = models
        return models


    def auto_layers(self,models,inpute,din,outpute, auto_compile = True, pw = None,num = 50):
        kiki = []
        man = 50
        if pw == 123456789:
            man = num
        if auto_compile:
            models.compile('adam',loss="mse",metrics=['accuracy'])
        inpute = int(inpute)
        outpute = int(outpute)
        models.add(ke.layers.Input(shape=[inpute]))
        for i in range(din):


            red = np.random.randint(20,man)
            kiki.append(red)
            models.add(ke.layers.Dense(red, tf.nn.relu))
            models.add(ke.layers.Dropout(0.4))

        models.add(ke.layers.Dropout(0.4))
        models.add(ke.layers.Dense(outpute))

        self._ANN_layers = kiki
        self.models = models

        return models

    def compile_ling(self,optimizer_= 'rmsprop',loss_="mse",metrics_=['accuracy']):
        models = self.models
        models.compile(optimizer= optimizer_ ,loss= loss_ ,metrics= metrics_ )
        self.models = models

    def layers_anti(self,printing = True):
        anss = self._ANN_layers
        if printing:
            print('\u001b[36m'+str(anss)+'\u001b[0m')
        return anss

    def input_models(self,model): #혹시 몰라 만듬
        self.models = model

    def tag_(x,x_size,y,y_size):

        pass

    def predict(self,x,printing = False, comparison = True):
        #print(x)
        anti = self.models.predict(x)
        anti_kiki = self.mond
        #print(anti_kiki)
        if printing:
            print(anti)
        if comparison:
            print(anti_kiki[int(anti)])

        return anti

