import pandas as pd
import numpy as np

da=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\car price 13 project ml\CarPrice_Assignment.csv")
print(da.head(4))
print(da.shape)
print(da.describe)
print(da.dtypes)
print(da.info)
print(da.values)


#data clining
print(da.isnull().sum())
s=da.drop(['car_ID'],axis=1)
#x and y finding\

x=da.select_dtypes(exclude=["object"])

print(x)

y=da.iloc[:,-1].values
print(y)


#scaling index varible

from sklearn.preprocessing import scale
co=x.columns.values
x=pd.DataFrame(scale(x))
x.columns=co
x

#split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#model train
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+")


#predict
pre=model.predict(x_test)
print(pre)
from sklearn.metrics import r2_score 
print("r2_score is {0}%".format(r2_score(pre,y_test)*100))

#for new car price
car_ID=int(input("enter ur'car_ID'like [9999]=:=")) # Assign a new unique ID for the new car
symboling=int(input("enter ur'symboling like [3,1,2,-1] =:="))
fueltype=input("enter ur'fueltype'like  ['gas','diesel] =:=")
aspiration=input("enter ur'aspiration'like  ['std','turbo',] =:=")
doornumber=input("enter ur'doornumber'like  ['four',two]=:=")
carbody=input("enter ur'carbody'like  ['sedan',convertible,hatchback,wagon]=:=")
drivewheel=input("enter urdrivewheel'like  ['fwd'rwd,]=:=")
enginelocation=input("enter ur'enginelocation'like  ['front',rear],=:=")
wheelbase=float(input("enter ur'wheelbase'like  100.2=:="))
carlength=float(input("enter ur'carlength'like  170.1=:="))
carwidth=float(input("enter ur'carwidth'like  65.1=:="))
carheight=float(input("enter ur'carheight'like  50.6=:="))
curbweight=float(input("enter ur'curbweigh like 2500.7=:="))
enginetype=input("enter ur'enginetype like ['ohc',ohcv,],=:=")
cylindernumber=input("enter ur 'cylindernumber'like ['four',six,two],=:=")
enginesize=float(input("enter ur'enginesize'like [120.78],=:="))
fuelsystem=input("enter ur'fuelsystem': ['mpfi',2bbl,mfi,1bbl,4bbl],=:=")
boreratio=float(input("enter ur 'boreratio' like [3.5],=:="))
stroke=float(input("enter ur'stroke'like [2.8],=:="))
compressionratio=float(input("enter ur 'compressionratio'like [8.0],=:="))
horsepower=float(input("enter ur'horsepower'like [100],=:="))
peakrpm=float(input("enter ur 'peakrpm'like [5000],=:="))
citympg=float(input("enter ur 'citympg'like [25],=:="))
highwaympg=float(input("enter ur'highwaympg'like [30]=:="))

new_car =  pd.DataFrame({	symboling,fueltype	,aspiration	,doornumber	,carbody,	
                drivewheel	,enginelocation	,wheelbase,
             	carlength	,carwidth,	carheight	,curbweight,
                enginetype	,cylindernumber,enginesize	,fuelsystem	,boreratio	,stroke	,compressionratio,horsepower	,peakrpm	,citympg	,highwaympg
})
new_car_encoded = pd.get_dummies(new_car)
new_car_scaled = pd.DataFrame(scale(new_car_encoded))
new_car_scaled.columns = new_car_encoded.columns

predicted_price = model.predict(new_car_scaled)
print('Predicted price for the new car is:', predicted_price[0])