import pandas as pd
from auto_tf_net import *
from matplotlib import pyplot

iris = pd.read_csv("https://raw.githubusercontent.com/Narehase/Bom/main/Iris.csv")

iris_y = iris[['Species']].values.tolist()
iris_x = iris[['PetalLengthCm','PetalWidthCm']].values.tolist()

ax_x = iris[['PetalLengthCm']].values.tolist()
ax_y = iris[['PetalWidthCm']].values.tolist()

co = []

ann = auto_tf_ANN()
named,y_axis = ann.list_label(iris_y)


for i in range(150):
    co.append(y_axis*10)

pyplot.scatter(ax_x,ax_y,c = y_axis)
pyplot.show()
#pyplot.show()

#x = [[9,1],[8,2],[7,3],[6,4],[5,5],[4,6],[3,7],[2,8],[1,9]]
x = [20,21,22,23]
y = [40,42,44,46]

ann.set_layers(auto_tf_ANN.keras_Sequential(),2,1,[20,280,500,400,200,100,10])
ann.callbak(limit_number=10)
ann.auto_acc(iris_x,y_axis,0.80,callbacks= True,limit= 100)
ann.predict([[20]],comparison=False,printing=True)