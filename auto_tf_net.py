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

class data_seting:

    def __init__(self):
        print('\u001b[32m'+"__edit_date__"+'\u001b[0m')
        self.on_and_off = False
        self.label_frame = []

    def list_label(self,list_):
        addon = []
        et = ''
        nu = -1
        for i in range(len(list_)):
            st = list_[i]

            if et != st:
                nu += 1
                addon.append(list_[i])

            et = list_[i]

            list_[i] = nu-1





        self.label_frame = addon
        self._list_ = list_
        self.on_and_off = True
        return addon, list_



        #print(list_)
        #print(addon)

        return list_ , addon


class auto_tf_ANN(data_seting):

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

    def auto_acc(self, x, y, acc= 1.00,  save = False, save_name = "auto_Model.h5", callbacks = False, limit = 10):

        models = self.models

        models = self.models
        stek = 0
        total = 0
        print(acc)
        while True:
            if callbacks:
                ack = self.call
                l = models.fit(x,y,epochs=100,callbacks=[ack])
                if len(l.history['loss']) < 99:
                    stek += 1

                if len(l.history['loss']) < 100 and stek == limit:
                    break


            else:
                l = models.fit(x,y,epochs=5)
            lossoff = l.history['accuracy']



            if acc == float(max(lossoff)):
                print(min(lossoff))
                break

        if save:
            models.save(save_name)

        self.models = models

        return models

    def auto_loss(self, x,y, loss = 0.01, save = False, save_name = "auto_Model.h5", callbacks = False, limit = 10):

        models = self.models
        stek = 0

        print("loss : ", loss)
        #print(models)
        while True:

            if callbacks:
                ack = self.call
                l = models.fit(x,y,epochs=100,callbacks=[ack])

                if len(l.history['loss']) < 99:
                    stek += 1

                if len(l.history['loss']) < 100 and stek == limit:
                    break
            else:
                l = models.fit(x,y,epochs=10)

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


    def callbak(self, monitoring = 'loss', limit_number = 3):
        clobac = ke.callbacks.EarlyStopping(monitor=monitoring,patience=limit_number)
        self.call = clobac
        models = self.models

        return models




    def set_layers(self,models,inpute,outpute,layer_out, auto_compile = True):
        print(layer_out)
        inpute = int(inpute)
        outpute = int(outpute)
        models.add(ke.layers.Input(shape=[inpute]))
        for i in range(len(layer_out)):
            models.add(ke.layers.Dense(layer_out[i], tf.nn.relu))
            models.add(ke.layers.Dropout(0.4))

        models.add(ke.layers.Dropout(0.4))
        models.add(ke.layers.Dense(outpute))


        if auto_compile:
            models.compile(loss='mse', optimizer='adam',metrics=['accuracy'])


        self._ANN_layers = layer_out
        self.models = models

        return models


    def auto_layers(self,models,inpute,din,outpute, auto_compile = True, pw = None,num = 50):
        kiki = []
        man = 50
        if pw == 123456789:
            man = num

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

        if auto_compile:
            models.compile('adam',loss="mse",metrics=['accuracy'])

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

    def predict(self,x,label = None,printing = False, comparison = True):
        #print(x)
        anti = self.models.predict(x)
        labels = self.on_and_off

        if label == None and not labels:
            anti_kiki = self.mond

        if labels:
            anti_kiki = self.label_frame

        #print(anti_kiki)
        if printing:
            print(anti)
        if comparison:
            print(anti_kiki[int(anti)])

        return anti


