import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
test=pd.read_csv('/home/tejas/ytest.csv',names=['price'])
test1=pd.read_csv('/home/tejas/ypred.csv',names=['price'])
test3=pd.read_csv('/home/tejas/final.csv')

from sklearn.datasets import load_boston
st.title('Welcome to Astrohouse')
import pandas as pd

@st.cache
def from_data_file(filename):
    url = (
        "https://raw.githubusercontent.com/streamlit/"
        "streamlit/develop/examples/data/%s" % filename)
    return pd.read_json(url)


ALL_LAYERS = {
    "View Map": {
        "type": "HexagonLayer",
        "data": from_data_file("bike_rental_stats.json"),
        "radius": 200,
        "elevationScale": 4,
        "elevationRange": [0, 1000],
        "pickable": True,
        "extruded": True,
    }

}



selected_layers = [layer for layer_name, layer in ALL_LAYERS.items()
    if st.sidebar.checkbox(layer_name, True)]

viewport={"latitude": 37.76, "longitude":  -122.4, "zoom": 11, "pitch": 50}
st.deck_gl_chart(viewport=viewport, layers=selected_layers)


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.dropna()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


a=test['price']
b=test1['price']
c=test3['RM']

if st.checkbox('Show dataframe'): 
     st.write(boston)


# col1 = st.selectbox('Which feature on x?', df.columns[0:4])
# col2 = st.selectbox('Which feature on y?', df.columns[0:4])new_df = df[(df['variety'].isin(species))]
# st.write(new_df)
# fig = px.scatter(boston,boston['LSTAT'],boston['MEDV'])
# st.plotly_chart(fig)

andy=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
values = st.sidebar.slider('Price range', 0, 55, (20, 30))
teams = st.sidebar.multiselect("Enter features", andy)
# st.write("Your input features", teams)
for i in range(len(teams)):
	andy.remove(teams[i])
# st.write(andy)
# f = px.histogram(boston.query(f'MEDV.between{values}'), x='price', nbins=15, title='Price distribution')
# f.update_xaxes(title='Price')
# f.update_yaxes(title='LSTAT')
# st.plotly_chart(f)
values=list(values)
st.write()
st.write('Enter number of rooms per dwelling you are looking for')
rm = st.text_input("")

st.write('Enter percentage of lower status of population')
lstat= st.text_input("Lower ratio is recommended")


if (lstat):


	with open('/home/tejas/model_pickle','rb') as f:
		mp=pickle.load(f)



	# abc=['lstat','rm']
	# abc=np.array(abc).reshape(-1,1)
	from sklearn.metrics import mean_squared_error

	rmse = (np.sqrt(mean_squared_error(a, b)))

	pr=mp.predict([[float(lstat),float(rm)]])
	x=pr[0]
	st.header(f'Predicted Price ${int(x*1000)}')
	st.subheader(f'The actual price may vary upto ${int(rmse*1000)}')


	import time
	latest_iteration = st.empty()
	bar = st.progress(0)



	if st.checkbox('View Graphical Data Dependency'): 
		for i in range(100):
	  # Update the progress bar with each iteration.
		  latest_iteration.text(f'Iteration {i+1}')
		  bar.progress(i + 1)
		  time.sleep(0.02)
	 

		plt.figure(figsize=(15, 5))

		features = ['LSTAT', 'RM']
		target = boston['MEDV']

		for i, col in enumerate(features):
		    plt.subplot(1, len(features) , i+1)
		    x = boston[col]
		    y = target
		    plt.scatter(x, y, marker='o')
		    plt.title(col)
		    plt.xlabel(col)
		    plt.ylabel('MEDV')
		    st.pyplot()

	if st.checkbox('Check Accuracy'): 

		for i in range(100):
	  # Update the progress bar with each iteration.
		  latest_iteration.text(f'Iteration {i+1}')
		  bar.progress(i + 1)
		  time.sleep(0.02)
		plt.title('Predicted VS Actual price')
		plt.figure(figsize=(11, 7))
		plt.scatter(c,a,color='blue',linewidth=3,label='Actual price')
		plt.scatter(c,b,color='red',linewidth=3,label='Predicted price')
		st.pyplot()






# l=boston['MEDV']

# for i in range(min(values),max(values)):
# 	if l<:
# 		st.write(i)

	if st.checkbox('View'): 

		# t=[]
		# for l in boston['MEDV']:
		# 	if l>min(values) and l<max(values):

				
		# 		# st.write(boston[boston['MEDV']==l])
		# 		t.append(l)
		# st.table(t)
		# r=boston['MEDV'].argmax[t]
		# st.write(r)
		
		x=(boston[(boston['MEDV']>min(values))&(boston['MEDV']<max(values))])
		x=x.drop(andy,axis=1)
		st.write(x)














