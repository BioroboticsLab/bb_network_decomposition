# bb_network_decomposition
Analyze social networks using spectral decomposition over time

```python
import datetime
import pandas as pd
import numpy as np

import bb_network_decomposition
import bb_network_decomposition.data
import bb_network_decomposition.normalization
import bb_network_decomposition.spectral
import bb_network_decomposition.projection
import bb_network_decomposition.evaluation

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
```


```python
raw_networks_path = "/home/ben/tmp/interaction_networks_20160812.h5"
locations_path = (
    "/home/ben/ssh/flip-storage/beesbook/circadians/bee_all_features.pickle"
)
alive_path = "/home/ben/ssh/flip-storage/beesbook/circadians/alive_bees_bayesian.csv"
```


```python
from_date = datetime.datetime(2016, 8, 12)
num_days = 1

(
    interactions,
    labels,
    bee_ids,
    bee_ages,
) = bb_network_decomposition.data.load_networks_h5(raw_networks_path, 0, num_days)
alive_df = bb_network_decomposition.data.load_alive_data(alive_path, bee_ids)

num_days = interactions.shape[0]
num_entities = interactions.shape[1]

num_modes = len(labels)
num_factors_per_mode = 8
```



```python
alive_matrices = bb_network_decomposition.data.get_daily_alive_matrices(
    alive_df, num_days, num_entities, from_date
)
alive_matrices.shape
```




    (1, 2010, 2010)



Boolean tensor containing lifetime data of every individual. Shape is Day x Inidividual x Individual.

If both individuals _i_ and _j_ were alive on day _d_, _alive_matrices\[d,i,j\]_ is True.


```python
interactions = bb_network_decomposition.normalization.rank_transform(
    interactions, alive_matrices
)
```


```python
interactions.shape
```




    (1, 2010, 2010, 9)



Interaction strenghts of individuals over time. Shape is Day x Individual x Individual x Interaction mode.


```python
labels
```




    ['proximity_counts',
     'proximity_euclidean',
     'proximity_rbf',
     'velocity_pos_sum',
     'velocity_neg_sum',
     'velocity_pos_mean',
     'velocity_neg_mean',
     'trophallaxis_duration',
     'trophallaxis_counts']



List of interaction modes in the same order as stored in _interactions_.


```python
(
    daily_factors,
    num_factors_by_mode,
) = bb_network_decomposition.spectral.decomposition_by_day(
    interactions, alive_matrices, num_factors_per_mode, num_jobs=4
)
```


```python
daily_factors[0].shape
```




    (2010, 104)



Spectral factors of interactions matrices over time before temporal alignment and CCA.


```python
num_factors = daily_factors[0].shape[-1]
```


```python
daily_factors_aligned = bb_network_decomposition.spectral.temporal_alignment(
    daily_factors, alive_matrices
)
```


```python
factor_df = bb_network_decomposition.data.get_factor_dataframe(
    daily_factors_aligned, from_date, alive_df, bee_ids
)
```


```python
factor_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day</th>
      <th>date</th>
      <th>bee_id</th>
      <th>age</th>
      <th>f_0</th>
      <th>f_1</th>
      <th>f_2</th>
      <th>f_3</th>
      <th>f_4</th>
      <th>f_5</th>
      <th>...</th>
      <th>f_94</th>
      <th>f_95</th>
      <th>f_96</th>
      <th>f_97</th>
      <th>f_98</th>
      <th>f_99</th>
      <th>f_100</th>
      <th>f_101</th>
      <th>f_102</th>
      <th>f_103</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>21</td>
      <td>45</td>
      <td>-0.0072421</td>
      <td>0.00314685</td>
      <td>-0.000766774</td>
      <td>0.00423605</td>
      <td>-0.000829805</td>
      <td>-0.00254063</td>
      <td>...</td>
      <td>0.0361333</td>
      <td>0.00422984</td>
      <td>0.00307967</td>
      <td>-0.000477117</td>
      <td>0.00108911</td>
      <td>0.000115448</td>
      <td>-0.000459577</td>
      <td>-0.000637633</td>
      <td>0.000243496</td>
      <td>0.000265616</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>39</td>
      <td>45</td>
      <td>-0.00492436</td>
      <td>-0.00218363</td>
      <td>-0.00100829</td>
      <td>0.00218637</td>
      <td>0.000999779</td>
      <td>0.00164859</td>
      <td>...</td>
      <td>0.0112397</td>
      <td>-0.0318492</td>
      <td>0.00347367</td>
      <td>-0.00029442</td>
      <td>0.00085713</td>
      <td>-3.16481e-05</td>
      <td>-0.000640272</td>
      <td>-0.000631403</td>
      <td>0.000944378</td>
      <td>0.00050027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>59</td>
      <td>45</td>
      <td>-0.00546352</td>
      <td>0.000335544</td>
      <td>-0.00101128</td>
      <td>0.00361958</td>
      <td>-0.00488807</td>
      <td>0.00168977</td>
      <td>...</td>
      <td>0.00228568</td>
      <td>0.00342571</td>
      <td>0.00108283</td>
      <td>0.000275634</td>
      <td>0.0004522</td>
      <td>0.00272719</td>
      <td>-0.00407326</td>
      <td>0.00188559</td>
      <td>0.00164427</td>
      <td>-0.00368969</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>178</td>
      <td>44</td>
      <td>-0.00129473</td>
      <td>-0.00042854</td>
      <td>-0.00443996</td>
      <td>0.00169473</td>
      <td>0.00609216</td>
      <td>0.00286463</td>
      <td>...</td>
      <td>0.00245098</td>
      <td>-0.00366821</td>
      <td>0.00191644</td>
      <td>0.000519615</td>
      <td>0.00236648</td>
      <td>-0.00175915</td>
      <td>-0.00159183</td>
      <td>0.000388643</td>
      <td>-0.00114168</td>
      <td>0.00312261</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>199</td>
      <td>44</td>
      <td>-0.00827621</td>
      <td>0.00464029</td>
      <td>-0.00073654</td>
      <td>0.00250572</td>
      <td>-0.0110911</td>
      <td>0.00379158</td>
      <td>...</td>
      <td>-0.011018</td>
      <td>-0.0115067</td>
      <td>0.00324116</td>
      <td>-1.78709e-05</td>
      <td>0.000811037</td>
      <td>-0.00177111</td>
      <td>-0.000170243</td>
      <td>0.000756058</td>
      <td>0.000523419</td>
      <td>0.000660676</td>
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
      <th>1198</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>3004</td>
      <td>1</td>
      <td>0.00563097</td>
      <td>0.00379659</td>
      <td>-0.00161652</td>
      <td>0.000676536</td>
      <td>-0.00236564</td>
      <td>-0.00542808</td>
      <td>...</td>
      <td>-0.00450846</td>
      <td>0.00490537</td>
      <td>-0.00153503</td>
      <td>0.0021592</td>
      <td>0.000309661</td>
      <td>0.00107967</td>
      <td>0.00028156</td>
      <td>-0.00251266</td>
      <td>0.0010452</td>
      <td>-0.00185304</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>3005</td>
      <td>1</td>
      <td>0.00482178</td>
      <td>0.00242558</td>
      <td>-0.00278712</td>
      <td>0.000206927</td>
      <td>-0.00144692</td>
      <td>-0.000719731</td>
      <td>...</td>
      <td>-0.000142806</td>
      <td>0.001626</td>
      <td>-0.00351581</td>
      <td>-0.00177971</td>
      <td>-0.00211917</td>
      <td>0.00397317</td>
      <td>-0.00144193</td>
      <td>-0.00410368</td>
      <td>0.00143616</td>
      <td>-0.0053216</td>
    </tr>
    <tr>
      <th>1200</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>3006</td>
      <td>1</td>
      <td>0.006286</td>
      <td>0.00478083</td>
      <td>-0.00603946</td>
      <td>-0.00192634</td>
      <td>-0.000801822</td>
      <td>-0.00546829</td>
      <td>...</td>
      <td>-0.00293883</td>
      <td>0.00529245</td>
      <td>-0.00371925</td>
      <td>0.00100774</td>
      <td>0.000242149</td>
      <td>0.00156555</td>
      <td>-0.003421</td>
      <td>-0.00081978</td>
      <td>-0.00020905</td>
      <td>0.000645418</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>3007</td>
      <td>1</td>
      <td>0.00502065</td>
      <td>0.00350009</td>
      <td>0.00158003</td>
      <td>0.000819389</td>
      <td>-8.73631e-05</td>
      <td>0.00113423</td>
      <td>...</td>
      <td>0.000790047</td>
      <td>-0.000764896</td>
      <td>-0.00439103</td>
      <td>0.000640282</td>
      <td>0.000883359</td>
      <td>0.00338366</td>
      <td>0.00165024</td>
      <td>0.00441386</td>
      <td>0.000327513</td>
      <td>-0.00464702</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>0</td>
      <td>2016-08-12</td>
      <td>3008</td>
      <td>1</td>
      <td>0.00536517</td>
      <td>0.00204567</td>
      <td>-0.00211659</td>
      <td>0.00075391</td>
      <td>0.000456557</td>
      <td>0.00402599</td>
      <td>...</td>
      <td>0.000324456</td>
      <td>0.00368738</td>
      <td>-0.00692263</td>
      <td>0.0016049</td>
      <td>-0.00120159</td>
      <td>-0.00108707</td>
      <td>0.00264493</td>
      <td>0.00237449</td>
      <td>0.0011709</td>
      <td>-0.0009587</td>
    </tr>
  </tbody>
</table>
<p>1203 rows × 108 columns</p>
</div>




```python
loc_df = bb_network_decomposition.data.load_location_data(locations_path)
```


```python
loc_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bee_id</th>
      <th>age</th>
      <th>brood_area</th>
      <th>brood_area_open</th>
      <th>brood_area_combined</th>
      <th>dance_floor</th>
      <th>honey_storage</th>
      <th>near_exit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2048</td>
      <td>21</td>
      <td>0.134021</td>
      <td>0.365979</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.474227</td>
      <td>0.025773</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2049</td>
      <td>21</td>
      <td>0.166667</td>
      <td>0.194444</td>
      <td>0.361111</td>
      <td>0.083333</td>
      <td>0.263889</td>
      <td>0.291667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2050</td>
      <td>21</td>
      <td>0.175549</td>
      <td>0.263323</td>
      <td>0.438871</td>
      <td>0.009404</td>
      <td>0.539185</td>
      <td>0.012539</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2051</td>
      <td>21</td>
      <td>0.484848</td>
      <td>0.363636</td>
      <td>0.848485</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.121212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2052</td>
      <td>21</td>
      <td>0.126582</td>
      <td>0.202532</td>
      <td>0.329114</td>
      <td>0.063291</td>
      <td>0.101266</td>
      <td>0.506329</td>
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
    </tr>
    <tr>
      <th>30979</th>
      <td>2012</td>
      <td>24</td>
      <td>0.617021</td>
      <td>0.085106</td>
      <td>0.702128</td>
      <td>0.012766</td>
      <td>0.165957</td>
      <td>0.119149</td>
    </tr>
    <tr>
      <th>30980</th>
      <td>2015</td>
      <td>24</td>
      <td>0.035398</td>
      <td>0.044248</td>
      <td>0.079646</td>
      <td>0.371681</td>
      <td>0.079646</td>
      <td>0.469027</td>
    </tr>
    <tr>
      <th>30981</th>
      <td>2017</td>
      <td>24</td>
      <td>0.000000</td>
      <td>0.002183</td>
      <td>0.002183</td>
      <td>0.480349</td>
      <td>0.000000</td>
      <td>0.517467</td>
    </tr>
    <tr>
      <th>30982</th>
      <td>2031</td>
      <td>24</td>
      <td>0.000000</td>
      <td>0.005917</td>
      <td>0.005917</td>
      <td>0.224852</td>
      <td>0.023669</td>
      <td>0.745562</td>
    </tr>
    <tr>
      <th>30983</th>
      <td>2033</td>
      <td>24</td>
      <td>0.560510</td>
      <td>0.210191</td>
      <td>0.770701</td>
      <td>0.038217</td>
      <td>0.127389</td>
      <td>0.063694</td>
    </tr>
  </tbody>
</table>
<p>26543 rows × 8 columns</p>
</div>




```python
cca_factor_df, cca = bb_network_decomposition.projection.get_cca_projection(
    factor_df, loc_df, return_cca=True, num_components=3
)
cca_factor_df.sort_values("date", inplace=True)
```


```python
cca_factor_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bee_id</th>
      <th>age</th>
      <th>brood_area</th>
      <th>brood_area_open</th>
      <th>brood_area_combined</th>
      <th>dance_floor</th>
      <th>honey_storage</th>
      <th>near_exit</th>
      <th>day</th>
      <th>date</th>
      <th>...</th>
      <th>f_102</th>
      <th>f_103</th>
      <th>network_age</th>
      <th>network_age_0</th>
      <th>network_age_1</th>
      <th>network_age_2</th>
      <th>location_age</th>
      <th>location_age_0</th>
      <th>location_age_1</th>
      <th>location_age_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2048</td>
      <td>18</td>
      <td>0.435644</td>
      <td>0.277228</td>
      <td>0.712871</td>
      <td>0.009901</td>
      <td>0.267327</td>
      <td>0.009901</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.000999945</td>
      <td>-0.00013982</td>
      <td>10.730456</td>
      <td>10.730456</td>
      <td>30.687013</td>
      <td>20.653116</td>
      <td>3.041829</td>
      <td>3.041829</td>
      <td>26.937071</td>
      <td>28.405706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2049</td>
      <td>18</td>
      <td>0.303571</td>
      <td>0.303571</td>
      <td>0.607143</td>
      <td>0.000000</td>
      <td>0.375000</td>
      <td>0.017857</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.00185249</td>
      <td>-0.00162587</td>
      <td>11.682501</td>
      <td>11.682501</td>
      <td>38.454187</td>
      <td>44.198147</td>
      <td>3.780845</td>
      <td>3.780845</td>
      <td>35.738029</td>
      <td>33.243674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2050</td>
      <td>18</td>
      <td>0.804979</td>
      <td>0.161826</td>
      <td>0.966805</td>
      <td>0.000000</td>
      <td>0.033195</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.0023183</td>
      <td>-0.00252885</td>
      <td>3.200392</td>
      <td>3.200392</td>
      <td>18.924773</td>
      <td>27.354647</td>
      <td>0.259778</td>
      <td>0.259778</td>
      <td>8.647924</td>
      <td>16.482956</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2051</td>
      <td>18</td>
      <td>0.327586</td>
      <td>0.534483</td>
      <td>0.862069</td>
      <td>0.000000</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>7.5265e-05</td>
      <td>0.00116405</td>
      <td>4.510731</td>
      <td>4.510731</td>
      <td>31.245857</td>
      <td>37.913376</td>
      <td>2.533463</td>
      <td>2.533463</td>
      <td>14.319041</td>
      <td>17.725555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2052</td>
      <td>18</td>
      <td>0.367347</td>
      <td>0.571429</td>
      <td>0.938776</td>
      <td>0.000000</td>
      <td>0.061224</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.000365201</td>
      <td>0.000409967</td>
      <td>3.545179</td>
      <td>3.545179</td>
      <td>22.177226</td>
      <td>24.799920</td>
      <td>0.496844</td>
      <td>0.496844</td>
      <td>10.825468</td>
      <td>17.897995</td>
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
      <th>1191</th>
      <td>2031</td>
      <td>17</td>
      <td>0.034483</td>
      <td>0.017241</td>
      <td>0.051724</td>
      <td>0.419540</td>
      <td>0.017241</td>
      <td>0.511494</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.00170887</td>
      <td>-0.000512209</td>
      <td>39.142523</td>
      <td>39.142523</td>
      <td>21.247015</td>
      <td>8.439924</td>
      <td>37.346153</td>
      <td>37.346153</td>
      <td>14.406179</td>
      <td>17.487104</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>2033</td>
      <td>17</td>
      <td>0.413793</td>
      <td>0.120690</td>
      <td>0.534483</td>
      <td>0.017241</td>
      <td>0.155172</td>
      <td>0.293103</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.00173122</td>
      <td>-0.00543476</td>
      <td>13.642365</td>
      <td>13.642365</td>
      <td>23.468995</td>
      <td>23.760079</td>
      <td>12.422779</td>
      <td>12.422779</td>
      <td>26.579519</td>
      <td>15.173826</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>895</td>
      <td>36</td>
      <td>0.028986</td>
      <td>0.000000</td>
      <td>0.028986</td>
      <td>0.260870</td>
      <td>0.014493</td>
      <td>0.695652</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>-0.00237317</td>
      <td>0.0018826</td>
      <td>30.448250</td>
      <td>30.448250</td>
      <td>24.907118</td>
      <td>18.065530</td>
      <td>36.571315</td>
      <td>36.571315</td>
      <td>22.881767</td>
      <td>6.197774</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>1995</td>
      <td>17</td>
      <td>0.044248</td>
      <td>0.035398</td>
      <td>0.079646</td>
      <td>0.415929</td>
      <td>0.008850</td>
      <td>0.495575</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>-0.000703707</td>
      <td>-0.000529077</td>
      <td>37.765633</td>
      <td>37.765633</td>
      <td>16.764451</td>
      <td>16.772350</td>
      <td>36.548395</td>
      <td>36.548395</td>
      <td>13.348284</td>
      <td>17.367083</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>2042</td>
      <td>17</td>
      <td>0.028226</td>
      <td>0.040323</td>
      <td>0.068548</td>
      <td>0.475806</td>
      <td>0.008065</td>
      <td>0.447581</td>
      <td>0</td>
      <td>2016-08-12</td>
      <td>...</td>
      <td>0.000554811</td>
      <td>0.000717112</td>
      <td>32.903401</td>
      <td>32.903401</td>
      <td>8.075877</td>
      <td>23.327540</td>
      <td>37.584135</td>
      <td>37.584135</td>
      <td>10.653747</td>
      <td>20.939378</td>
    </tr>
  </tbody>
</table>
<p>1197 rows × 122 columns</p>
</div>


factor_df.to_csv('/home/ben/ssh/flip-storage/beesbook/circadians/network_age_cca_v6.csv')

```python
variable_names = [
    ["age"],
    ["age", "network_age"],
    ["network_age"],
    ["network_age", "network_age_1"],
    ["network_age", "network_age_1", "network_age_2"],
]

targets = [bb_network_decomposition.constants.supplementary_labels] + list(
    map(lambda l: [l], bb_network_decomposition.constants.supplementary_labels)
)

target_cols = bb_network_decomposition.constants.supplementary_labels
```


```python
sup_df = bb_network_decomposition.data.load_supplementary_data(
    locations_path,
    keepcols=bb_network_decomposition.constants.default_location_data_cols
    + bb_network_decomposition.constants.default_supplementary_data_cols
    + ["location_descriptor_count"],
)
```


```python
location_cols = set(bb_network_decomposition.constants.location_labels).union(
    set(bb_network_decomposition.constants.location_cols)
)
```


```python
cca_factor_df = cca_factor_df[
    [c for c in cca_factor_df.columns if c not in location_cols]
]
```


```python
sup_df = bb_network_decomposition.data.merge_location_data(cca_factor_df, sup_df)
```


```python
sup_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bee_id</th>
      <th>age</th>
      <th>brood_area</th>
      <th>brood_area_open</th>
      <th>brood_area_combined</th>
      <th>dance_floor</th>
      <th>honey_storage</th>
      <th>near_exit</th>
      <th>r_squared</th>
      <th>day_activity</th>
      <th>...</th>
      <th>f_102</th>
      <th>f_103</th>
      <th>network_age</th>
      <th>network_age_0</th>
      <th>network_age_1</th>
      <th>network_age_2</th>
      <th>location_age</th>
      <th>location_age_0</th>
      <th>location_age_1</th>
      <th>location_age_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2048</td>
      <td>18</td>
      <td>0.435644</td>
      <td>0.277228</td>
      <td>0.712871</td>
      <td>0.009901</td>
      <td>0.267327</td>
      <td>0.009901</td>
      <td>0.013240</td>
      <td>0.159450</td>
      <td>...</td>
      <td>0.000999945</td>
      <td>-0.00013982</td>
      <td>10.730456</td>
      <td>10.730456</td>
      <td>30.687013</td>
      <td>20.653116</td>
      <td>3.041829</td>
      <td>3.041829</td>
      <td>26.937071</td>
      <td>28.405706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2049</td>
      <td>18</td>
      <td>0.303571</td>
      <td>0.303571</td>
      <td>0.607143</td>
      <td>0.000000</td>
      <td>0.375000</td>
      <td>0.017857</td>
      <td>0.024495</td>
      <td>0.541923</td>
      <td>...</td>
      <td>0.00185249</td>
      <td>-0.00162587</td>
      <td>11.682501</td>
      <td>11.682501</td>
      <td>38.454187</td>
      <td>44.198147</td>
      <td>3.780845</td>
      <td>3.780845</td>
      <td>35.738029</td>
      <td>33.243674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2050</td>
      <td>18</td>
      <td>0.804979</td>
      <td>0.161826</td>
      <td>0.966805</td>
      <td>0.000000</td>
      <td>0.033195</td>
      <td>0.000000</td>
      <td>0.064151</td>
      <td>0.294336</td>
      <td>...</td>
      <td>0.0023183</td>
      <td>-0.00252885</td>
      <td>3.200392</td>
      <td>3.200392</td>
      <td>18.924773</td>
      <td>27.354647</td>
      <td>0.259778</td>
      <td>0.259778</td>
      <td>8.647924</td>
      <td>16.482956</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2051</td>
      <td>18</td>
      <td>0.327586</td>
      <td>0.534483</td>
      <td>0.862069</td>
      <td>0.000000</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0.041037</td>
      <td>0.773193</td>
      <td>...</td>
      <td>7.5265e-05</td>
      <td>0.00116405</td>
      <td>4.510731</td>
      <td>4.510731</td>
      <td>31.245857</td>
      <td>37.913376</td>
      <td>2.533463</td>
      <td>2.533463</td>
      <td>14.319041</td>
      <td>17.725555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2052</td>
      <td>18</td>
      <td>0.367347</td>
      <td>0.571429</td>
      <td>0.938776</td>
      <td>0.000000</td>
      <td>0.061224</td>
      <td>0.000000</td>
      <td>0.062497</td>
      <td>1.245740</td>
      <td>...</td>
      <td>0.000365201</td>
      <td>0.000409967</td>
      <td>3.545179</td>
      <td>3.545179</td>
      <td>22.177226</td>
      <td>24.799920</td>
      <td>0.496844</td>
      <td>0.496844</td>
      <td>10.825468</td>
      <td>17.897995</td>
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
      <th>1186</th>
      <td>2033</td>
      <td>17</td>
      <td>0.413793</td>
      <td>0.120690</td>
      <td>0.534483</td>
      <td>0.017241</td>
      <td>0.155172</td>
      <td>0.293103</td>
      <td>0.001746</td>
      <td>0.074340</td>
      <td>...</td>
      <td>0.00173122</td>
      <td>-0.00543476</td>
      <td>13.642365</td>
      <td>13.642365</td>
      <td>23.468995</td>
      <td>23.760079</td>
      <td>12.422779</td>
      <td>12.422779</td>
      <td>26.579519</td>
      <td>15.173826</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>895</td>
      <td>36</td>
      <td>0.028986</td>
      <td>0.000000</td>
      <td>0.028986</td>
      <td>0.260870</td>
      <td>0.014493</td>
      <td>0.695652</td>
      <td>0.008698</td>
      <td>0.405214</td>
      <td>...</td>
      <td>-0.00237317</td>
      <td>0.0018826</td>
      <td>30.448250</td>
      <td>30.448250</td>
      <td>24.907118</td>
      <td>18.065530</td>
      <td>36.571315</td>
      <td>36.571315</td>
      <td>22.881767</td>
      <td>6.197774</td>
    </tr>
    <tr>
      <th>1188</th>
      <td>2037</td>
      <td>17</td>
      <td>0.088106</td>
      <td>0.022026</td>
      <td>0.110132</td>
      <td>0.801762</td>
      <td>0.013216</td>
      <td>0.074890</td>
      <td>0.171487</td>
      <td>2.053916</td>
      <td>...</td>
      <td>0.00153118</td>
      <td>-0.00209033</td>
      <td>32.492683</td>
      <td>32.492683</td>
      <td>2.392841</td>
      <td>32.826693</td>
      <td>39.370128</td>
      <td>39.370128</td>
      <td>-6.627234</td>
      <td>43.950088</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>2040</td>
      <td>17</td>
      <td>0.006173</td>
      <td>0.012346</td>
      <td>0.018519</td>
      <td>0.253086</td>
      <td>0.006173</td>
      <td>0.722222</td>
      <td>0.209580</td>
      <td>1.805024</td>
      <td>...</td>
      <td>0.00233186</td>
      <td>-0.00113773</td>
      <td>41.503666</td>
      <td>41.503666</td>
      <td>20.560036</td>
      <td>-5.679204</td>
      <td>37.082943</td>
      <td>37.082943</td>
      <td>23.188080</td>
      <td>4.744723</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>2042</td>
      <td>17</td>
      <td>0.028226</td>
      <td>0.040323</td>
      <td>0.068548</td>
      <td>0.475806</td>
      <td>0.008065</td>
      <td>0.447581</td>
      <td>0.105858</td>
      <td>1.129854</td>
      <td>...</td>
      <td>0.000554811</td>
      <td>0.000717112</td>
      <td>32.903401</td>
      <td>32.903401</td>
      <td>8.075877</td>
      <td>23.327540</td>
      <td>37.584135</td>
      <td>37.584135</td>
      <td>10.653747</td>
      <td>20.939378</td>
    </tr>
  </tbody>
</table>
<p>1191 rows × 131 columns</p>
</div>

```python
regression_results = bb_network_decomposition.evaluation.get_bootstrap_results(
    sup_df,
    variable_names,
    targets,
    regression=True,
    use_tqdm=True,
    num_bootstrap_samples=2,
)
```

```python
regression_results.groupby(["predictors", "target"]).fitted_linear_r2.mean()
```




    predictors                               target                                                                               
    age                                      amplitude                                                                                0.419735
                                             day_activity                                                                             0.335031
                                             days_left                                                                                0.010323
                                             phase                                                                                    0.018507
                                             r_squared                                                                                0.343281
                                             r_squared,day_activity,phase,amplitude,days_left,velocity,velocity_day,velocity_night    0.196286
                                             velocity                                                                                 0.003413
                                             velocity_day                                                                             0.079846
                                             velocity_night                                                                           0.287958
    age,network_age                          amplitude                                                                                0.725977
                                             day_activity                                                                             0.602565
                                             days_left                                                                                0.163612
                                             phase                                                                                    0.032984
                                             r_squared                                                                                0.657478
                                             r_squared,day_activity,phase,amplitude,days_left,velocity,velocity_day,velocity_night    0.356304
                                             velocity                                                                                 0.053733
                                             velocity_day                                                                             0.315265
                                             velocity_night                                                                           0.317176
    network_age                              amplitude                                                                                0.712313
                                             day_activity                                                                             0.592811
                                             days_left                                                                                0.130533
                                             phase                                                                                    0.029762
                                             r_squared                                                                                0.705417
                                             r_squared,day_activity,phase,amplitude,days_left,velocity,velocity_day,velocity_night    0.330845
                                             velocity                                                                                 0.015361
                                             velocity_day                                                                             0.276245
                                             velocity_night                                                                           0.287315
    network_age,network_age_1                amplitude                                                                                0.744016
                                             day_activity                                                                             0.603495
                                             days_left                                                                                0.218985
                                             phase                                                                                    0.030739
                                             r_squared                                                                                0.697034
                                             r_squared,day_activity,phase,amplitude,days_left,velocity,velocity_day,velocity_night    0.438993
                                             velocity                                                                                 0.279188
                                             velocity_day                                                                             0.467277
                                             velocity_night                                                                           0.418830
    network_age,network_age_1,network_age_2  amplitude                                                                                0.739748
                                             day_activity                                                                             0.617271
                                             days_left                                                                                0.284631
                                             phase                                                                                    0.040759
                                             r_squared                                                                                0.686936
                                             r_squared,day_activity,phase,amplitude,days_left,velocity,velocity_day,velocity_night    0.435070
                                             velocity                                                                                 0.312499
                                             velocity_day                                                                             0.477982
                                             velocity_night                                                                           0.439866
    Name: fitted_linear_r2, dtype: float64




```python
regression_results = bb_network_decomposition.evaluation.get_bootstrap_results(
    sup_df, variable_names, regression=False, use_tqdm=True, num_bootstrap_samples=2,
)
```

```python
regression_results.groupby(["predictors", "target"]).rho_mcf_linear.mean()
```




    predictors                               target                                                 
    age                                      brood_area_combined                                        0.557943
                                             dance_floor                                                0.433755
                                             dance_floor,honey_storage,brood_area_combined,near_exit    0.413200
                                             honey_storage                                              0.026341
                                             near_exit                                                  0.294098
    age,network_age                          brood_area_combined                                        0.835141
                                             dance_floor                                                0.690013
                                             dance_floor,honey_storage,brood_area_combined,near_exit    0.708147
                                             honey_storage                                              0.137835
                                             near_exit                                                  0.602501
    network_age                              brood_area_combined                                        0.827348
                                             dance_floor                                                0.652492
                                             dance_floor,honey_storage,brood_area_combined,near_exit    0.686810
                                             honey_storage                                              0.009662
                                             near_exit                                                  0.577548
    network_age,network_age_1                brood_area_combined                                        0.906312
                                             dance_floor                                                0.770685
                                             dance_floor,honey_storage,brood_area_combined,near_exit    0.842926
                                             honey_storage                                              0.667625
                                             near_exit                                                  0.747052
    network_age,network_age_1,network_age_2  brood_area_combined                                        0.919979
                                             dance_floor                                                0.801625
                                             dance_floor,honey_storage,brood_area_combined,near_exit    0.852838
                                             honey_storage                                              0.690189
                                             near_exit                                                  0.756670
    Name: rho_mcf_linear, dtype: float64
