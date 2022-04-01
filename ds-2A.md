```python
#Team 2A: Wei Wang, Zhang Haoran,Shane Reid
```


```python
#import all the library we need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols
#import the WineSales dataset
Sales=pd.read_csv("WineSales.csv")
Sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>nps</th>
      <th>marketing_spend</th>
      <th>products</th>
      <th>week</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$1,068</td>
      <td>4.8</td>
      <td>$442</td>
      <td>wine 2</td>
      <td>2015/1/4</td>
      <td>20151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$992</td>
      <td>4.7</td>
      <td>$442</td>
      <td>wine 2</td>
      <td>2015/1/11</td>
      <td>20151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$1,025</td>
      <td>5.5</td>
      <td>$442</td>
      <td>wine 2</td>
      <td>2015/1/18</td>
      <td>20151</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$1,030</td>
      <td>5.0</td>
      <td>$442</td>
      <td>wine 2</td>
      <td>2015/1/25</td>
      <td>20151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$850</td>
      <td>4.6</td>
      <td>$442</td>
      <td>wine 2</td>
      <td>2015/2/1</td>
      <td>20151</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>$2,917</td>
      <td>8.2</td>
      <td>$234</td>
      <td>wine 3</td>
      <td>2019/9/1</td>
      <td>20193</td>
    </tr>
    <tr>
      <th>244</th>
      <td>$3,218</td>
      <td>8.3</td>
      <td>$234</td>
      <td>wine 3</td>
      <td>2019/9/8</td>
      <td>20193</td>
    </tr>
    <tr>
      <th>245</th>
      <td>$2,847</td>
      <td>6.8</td>
      <td>$234</td>
      <td>wine 3</td>
      <td>2019/9/15</td>
      <td>20193</td>
    </tr>
    <tr>
      <th>246</th>
      <td>$3,292</td>
      <td>7.8</td>
      <td>$234</td>
      <td>wine 3</td>
      <td>2019/9/22</td>
      <td>20193</td>
    </tr>
    <tr>
      <th>247</th>
      <td>$2,999</td>
      <td>7.8</td>
      <td>$234</td>
      <td>wine 3</td>
      <td>2019/9/29</td>
      <td>20193</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 6 columns</p>
</div>




```python
#GOAL 1.1 Impute missing values
#There are 34 missing values in nps
print("Missing value before interpolate\n",Sales.isnull().sum())
#We use linear interpolate method to impute missing values
Sales['nps']=Sales['nps'].interpolate()
print("\nNo Missing value after interpolate\n",Sales.isnull().sum())
#We can see from below that all missing values are imputed.
```

    Missing value before interpolate
     sales               0
    nps                34
    marketing_spend     0
    products            0
    week                0
    quarter             0
    dtype: int64
    
    No Missing value after interpolate
     sales              0
    nps                0
    marketing_spend    0
    products           0
    week               0
    quarter            0
    dtype: int64
    


```python
#GOAL 1.2 one-hot encode products category
Sales.replace("wine 1,wine 3,wine 2","wine 1,wine 2,wine 3",inplace=True)
Sales.replace("wine 2,wine 1,wine 3","wine 1,wine 2,wine 3",inplace=True)
Sales.replace("wine 3,wine 1","wine 1,wine 3",inplace=True)
Sales.replace("wine 2","wine2",inplace=True)

#Use get_dummies method to one-hot encode the products category.
OneHotEncode= pd.get_dummies(Sales.products)
## delete the previous products column
Sales.drop('products',inplace=True,axis=1)
## add the onehotencode to the table
Sales=Sales.join(OneHotEncode)
Sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>nps</th>
      <th>marketing_spend</th>
      <th>week</th>
      <th>quarter</th>
      <th>wine 1,wine 2,wine 3</th>
      <th>wine 1,wine 3</th>
      <th>wine 2,wine 1</th>
      <th>wine 2,wine 3</th>
      <th>wine 3</th>
      <th>wine2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$1,068</td>
      <td>4.8</td>
      <td>$442</td>
      <td>2015/1/4</td>
      <td>20151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$992</td>
      <td>4.7</td>
      <td>$442</td>
      <td>2015/1/11</td>
      <td>20151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$1,025</td>
      <td>5.5</td>
      <td>$442</td>
      <td>2015/1/18</td>
      <td>20151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$1,030</td>
      <td>5.0</td>
      <td>$442</td>
      <td>2015/1/25</td>
      <td>20151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$850</td>
      <td>4.6</td>
      <td>$442</td>
      <td>2015/2/1</td>
      <td>20151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>$2,917</td>
      <td>8.2</td>
      <td>$234</td>
      <td>2019/9/1</td>
      <td>20193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>244</th>
      <td>$3,218</td>
      <td>8.3</td>
      <td>$234</td>
      <td>2019/9/8</td>
      <td>20193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>$2,847</td>
      <td>6.8</td>
      <td>$234</td>
      <td>2019/9/15</td>
      <td>20193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>$3,292</td>
      <td>7.8</td>
      <td>$234</td>
      <td>2019/9/22</td>
      <td>20193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>247</th>
      <td>$2,999</td>
      <td>7.8</td>
      <td>$234</td>
      <td>2019/9/29</td>
      <td>20193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 11 columns</p>
</div>




```python
# GOAL 2.1 Aggregate features by quarter
Sales['sales'] = Sales['sales'].map(lambda x: str(x)[1:])
Sales['sales'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
Sales=Sales.astype({"sales":"float"})
Sales['marketing_spend'] = Sales['marketing_spend'].map(lambda x: str(x)[1:])
Sales['marketing_spend'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
Sales=Sales.astype({"marketing_spend":"float"})

#Aggregate features by quarter and then we get total quarter sales, nps, marketing_spend, and wines products
Sales = Sales.groupby(['quarter']).agg({"sales":"sum",'nps':'sum','marketing_spend':'sum','wine 1,wine 2,wine 3':'sum','wine 1,wine 3':'sum','wine2':'sum','wine 2,wine 1':'sum','wine 2,wine 3':'sum','wine 3':'sum'})
Sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>nps</th>
      <th>marketing_spend</th>
      <th>wine 1,wine 2,wine 3</th>
      <th>wine 1,wine 3</th>
      <th>wine2</th>
      <th>wine 2,wine 1</th>
      <th>wine 2,wine 3</th>
      <th>wine 3</th>
    </tr>
    <tr>
      <th>quarter</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20151</th>
      <td>13330.0</td>
      <td>65.35</td>
      <td>5746.0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20152</th>
      <td>46022.0</td>
      <td>64.15</td>
      <td>1664.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20153</th>
      <td>35934.0</td>
      <td>66.30</td>
      <td>2119.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20154</th>
      <td>28126.0</td>
      <td>63.20</td>
      <td>3874.0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20161</th>
      <td>35362.0</td>
      <td>59.60</td>
      <td>4992.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20162</th>
      <td>44154.0</td>
      <td>64.65</td>
      <td>481.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20163</th>
      <td>34924.0</td>
      <td>64.10</td>
      <td>273.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20164</th>
      <td>36365.0</td>
      <td>63.60</td>
      <td>5135.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20171</th>
      <td>30554.0</td>
      <td>65.95</td>
      <td>845.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20172</th>
      <td>39667.0</td>
      <td>70.25</td>
      <td>7306.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20173</th>
      <td>47186.0</td>
      <td>78.65</td>
      <td>845.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20174</th>
      <td>41857.0</td>
      <td>105.35</td>
      <td>3696.0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20181</th>
      <td>36054.0</td>
      <td>95.25</td>
      <td>6648.0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20182</th>
      <td>51559.0</td>
      <td>105.00</td>
      <td>7631.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20183</th>
      <td>54924.0</td>
      <td>110.50</td>
      <td>1344.0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20184</th>
      <td>41654.0</td>
      <td>105.10</td>
      <td>4147.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20191</th>
      <td>39984.0</td>
      <td>101.90</td>
      <td>4888.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20192</th>
      <td>49520.0</td>
      <td>102.90</td>
      <td>5265.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20193</th>
      <td>45107.0</td>
      <td>101.45</td>
      <td>3042.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GOAL 2.2
#shift these features with quarter sales
Sales=Sales.assign(nps=lambda x: x.nps.shift(1).values,
marketing_spend=lambda x: x.marketing_spend.shift(1).values)
Sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>nps</th>
      <th>marketing_spend</th>
      <th>wine 1,wine 2,wine 3</th>
      <th>wine 1,wine 3</th>
      <th>wine2</th>
      <th>wine 2,wine 1</th>
      <th>wine 2,wine 3</th>
      <th>wine 3</th>
    </tr>
    <tr>
      <th>quarter</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20151</th>
      <td>13330.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20152</th>
      <td>46022.0</td>
      <td>65.35</td>
      <td>5746.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20153</th>
      <td>35934.0</td>
      <td>64.15</td>
      <td>1664.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20154</th>
      <td>28126.0</td>
      <td>66.30</td>
      <td>2119.0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20161</th>
      <td>35362.0</td>
      <td>63.20</td>
      <td>3874.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20162</th>
      <td>44154.0</td>
      <td>59.60</td>
      <td>4992.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20163</th>
      <td>34924.0</td>
      <td>64.65</td>
      <td>481.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20164</th>
      <td>36365.0</td>
      <td>64.10</td>
      <td>273.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20171</th>
      <td>30554.0</td>
      <td>63.60</td>
      <td>5135.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20172</th>
      <td>39667.0</td>
      <td>65.95</td>
      <td>845.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20173</th>
      <td>47186.0</td>
      <td>70.25</td>
      <td>7306.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20174</th>
      <td>41857.0</td>
      <td>78.65</td>
      <td>845.0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20181</th>
      <td>36054.0</td>
      <td>105.35</td>
      <td>3696.0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20182</th>
      <td>51559.0</td>
      <td>95.25</td>
      <td>6648.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20183</th>
      <td>54924.0</td>
      <td>105.00</td>
      <td>7631.0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20184</th>
      <td>41654.0</td>
      <td>110.50</td>
      <td>1344.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20191</th>
      <td>39984.0</td>
      <td>105.10</td>
      <td>4147.0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20192</th>
      <td>49520.0</td>
      <td>101.90</td>
      <td>4888.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20193</th>
      <td>45107.0</td>
      <td>102.90</td>
      <td>5265.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GOAL 2.3 Find correlation between features
Sales.corr()
#nps,marketing_spend and wine 2 has a correlation absolute value > 0.4
#We use these three variables to fit the multi-regression model
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sales</th>
      <th>nps</th>
      <th>marketing_spend</th>
      <th>wine 1,wine 2,wine 3</th>
      <th>wine 1,wine 3</th>
      <th>wine2</th>
      <th>wine 2,wine 1</th>
      <th>wine 2,wine 3</th>
      <th>wine 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sales</th>
      <td>1.000000</td>
      <td>0.487498</td>
      <td>0.633842</td>
      <td>0.294349</td>
      <td>0.332046</td>
      <td>-0.701769</td>
      <td>-0.231075</td>
      <td>-0.154923</td>
      <td>0.140926</td>
    </tr>
    <tr>
      <th>nps</th>
      <td>0.487498</td>
      <td>1.000000</td>
      <td>0.284025</td>
      <td>0.255818</td>
      <td>-0.018123</td>
      <td>-0.183782</td>
      <td>-0.218348</td>
      <td>-0.303365</td>
      <td>0.284777</td>
    </tr>
    <tr>
      <th>marketing_spend</th>
      <td>0.633842</td>
      <td>0.284025</td>
      <td>1.000000</td>
      <td>-0.188336</td>
      <td>0.430298</td>
      <td>-0.164598</td>
      <td>0.146133</td>
      <td>-0.397111</td>
      <td>0.159527</td>
    </tr>
    <tr>
      <th>wine 1,wine 2,wine 3</th>
      <td>0.294349</td>
      <td>0.255818</td>
      <td>-0.188336</td>
      <td>1.000000</td>
      <td>-0.508841</td>
      <td>-0.292240</td>
      <td>-0.200822</td>
      <td>-0.292240</td>
      <td>-0.200822</td>
    </tr>
    <tr>
      <th>wine 1,wine 3</th>
      <td>0.332046</td>
      <td>-0.018123</td>
      <td>0.430298</td>
      <td>-0.508841</td>
      <td>1.000000</td>
      <td>-0.204844</td>
      <td>-0.140766</td>
      <td>-0.204844</td>
      <td>-0.140766</td>
    </tr>
    <tr>
      <th>wine2</th>
      <td>-0.701769</td>
      <td>-0.183782</td>
      <td>-0.164598</td>
      <td>-0.292240</td>
      <td>-0.204844</td>
      <td>1.000000</td>
      <td>-0.080845</td>
      <td>-0.117647</td>
      <td>-0.080845</td>
    </tr>
    <tr>
      <th>wine 2,wine 1</th>
      <td>-0.231075</td>
      <td>-0.218348</td>
      <td>0.146133</td>
      <td>-0.200822</td>
      <td>-0.140766</td>
      <td>-0.080845</td>
      <td>1.000000</td>
      <td>-0.080845</td>
      <td>-0.055556</td>
    </tr>
    <tr>
      <th>wine 2,wine 3</th>
      <td>-0.154923</td>
      <td>-0.303365</td>
      <td>-0.397111</td>
      <td>-0.292240</td>
      <td>-0.204844</td>
      <td>-0.117647</td>
      <td>-0.080845</td>
      <td>1.000000</td>
      <td>-0.080845</td>
    </tr>
    <tr>
      <th>wine 3</th>
      <td>0.140926</td>
      <td>0.284777</td>
      <td>0.159527</td>
      <td>-0.200822</td>
      <td>-0.140766</td>
      <td>-0.080845</td>
      <td>-0.055556</td>
      <td>-0.080845</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GOAL 3.1 build multi-regression model using nps,marketing_spend and wine 2 as the independent variables, sales as dependent variable.
model=ols("sales~nps+marketing_spend+wine2", Sales).fit()
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  sales   R-squared:                       0.597
    Model:                            OLS   Adj. R-squared:                  0.510
    Method:                 Least Squares   F-statistic:                     6.908
    Date:                Fri, 01 Apr 2022   Prob (F-statistic):            0.00438
    Time:                        13:07:25   Log-Likelihood:                -176.79
    No. Observations:                  18   AIC:                             361.6
    Df Residuals:                      14   BIC:                             365.1
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept        2.742e+04   5391.733      5.086      0.000    1.59e+04     3.9e+04
    nps               106.7628     66.295      1.610      0.130     -35.427     248.952
    marketing_spend     1.4937      0.532      2.809      0.014       0.353       2.634
    wine2            -733.6809    409.996     -1.789      0.095   -1613.034     145.673
    ==============================================================================
    Omnibus:                        5.641   Durbin-Watson:                   2.775
    Prob(Omnibus):                  0.060   Jarque-Bera (JB):                3.624
    Skew:                          -1.085   Prob(JB):                        0.163
    Kurtosis:                       3.353   Cond. No.                     1.99e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.99e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

    C:\Users\Lenovo\anaconda3\lib\site-packages\scipy\stats\stats.py:1603: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=18
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "
    


```python
# GOAL 3.2 Predict 2019 Q4 sales
q4nps=sum(Sales['nps'][16:19])/3
q4mspend=sum(Sales['marketing_spend'][16:19])/3
q4wine2=sum(Sales['wine2'][16:19])/3
results=model.predict({"nps":q4nps, "marketing_spend":q4mspend, "wine2":q4wine2})
```


```python
#The final prediction for Q4 sales is:
results
```




    0    45568.866671
    dtype: float64


