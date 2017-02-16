from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot as plt
import os
import numpy as np
# from IPython.display import Image
# import plotly.plotly as py
# import plotly.graph_objs as go
# py.sign_in('ivychen', 'Gol0AP8lJ2GVI7FTaeI7')

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
# x = ['a','b','c','d','e','f','g','h']

twenty_all = fetch_20newsgroups(subset='all', categories=categories)
y = [0] * len(categories)

for i in range(len(twenty_all.data)):
	temp = categories.index(twenty_all.target_names[twenty_all.target[i]])
	y[temp] = y[temp] + 1

y_pos = np.arange(len(categories))
#plot histogram
if not os.path.exists('../Graphs/part_A'):
	os.makedirs('../Graphs/part_A')
plt.figure(1)
plt.bar(y_pos, y, align='center')
plt.xticks(y_pos, categories, rotation=23)

plt.xlabel("sub_class")
plt.ylabel("Count")
plt.title('number of document per topic')
plt.savefig('../Graphs/part_A/hist.png')

# data = [go.Bar(x=categories, y=y)]
# layout = go.Layout(title='number of document per topic', xaxis=dict(title="sub_class"))

# if not os.path.exists('../Graphs/part_A'):
# 	os.makedirs('../Graphs/part_A')

# fig = go.Figure(data=data, layout=layout)
# py.image.save_as(fig, filename='../Graphs/part_A/hist.jpeg')

N_comp = sum(y[0:4])
N_rec = sum(y[4:8])
print ("The number of documents in Computer Techonology is " + str(N_comp))
print ("The number of documents in Recreational Activity is " + str(N_rec))
