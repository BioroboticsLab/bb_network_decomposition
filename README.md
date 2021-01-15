Note: A rendered version of this markdown readme file can be found here: github.com/BioroboticsLab/bb_network_decomposition

# Social networks predict the life and death of honey bees
Analyze social networks using spectral decomposition over time.

Preprint: [DOI 10.1101/2020.05.06.076943](https://doi.org/10.1101/2020.05.06.076943 )
Data: [DOI 10.5281/zenodo.4438013](https://doi.org/10.5281/zenodo.4438013 )

## Usage example

This sample code showcases how to load the raw input data, calculate network age, and fit and evaluate the multinomial task regression and supplementary regression models.

```python
import datetime
import pandas as pd
import numpy as np

# https://github.com/BioroboticsLab/bb_network_decomposition

# the module can be installed using pip:
# $ pip3 install --user git+https://github.com/BioroboticsLab/bb_network_decomposition.git
# the dependencies should be installed automatically:
# https://github.com/BioroboticsLab/bb_network_decomposition/blob/master/requirements.txt
# please note that you may have to install the dependency bb_utils manually:
# $ pip3 install --user git+https://github.com/BioroboticsLab/bb_utils.git
import bb_network_decomposition
import bb_network_decomposition.data
import bb_network_decomposition.normalization
import bb_network_decomposition.spectral
import bb_network_decomposition.projection
import bb_network_decomposition.evaluation
```


```python
# location of interaction network hdf5 file
raw_networks_path = "zenodo/interaction_networks_20160729to20160827.h5"

# location of bee metainfo (location descriptors, supplementary labels, ...)
supplementary_data_path = "zenodo/bee_daily_data.csv"

# location of results of bayesian lifetime model
alive_path = "zenodo/alive_bees_bayesian.csv"
```


```python
# first date in the interaction tensor
# used to match interaction data with supplementary data (locations, etc.)
from_date = datetime.datetime(2016, 8, 12)

# number of days to use (incrase to reproduce paper results)
num_days = 1

# load interaction data
(
    interactions, # interaction tensor
    labels, # names of interaction modes (proximity, trophallaxis, etc.)
    bee_ids, # unique BeesBook IDs of the individuals
    bee_ages, # tensor with ages of individuals over time
) = bb_network_decomposition.data.load_networks_h5(raw_networks_path, 0, num_days)

alive_df = bb_network_decomposition.data.load_alive_data(alive_path, bee_ids)

num_days = interactions.shape[0]
num_entities = interactions.shape[1]

num_modes = len(labels)
# number of spectral factors per interaction mode
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

Spectral factors of interactions matrices over time after temporal alignment without CCA projection.


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
      <td>-0.00423605</td>
      <td>0.00082973</td>
      <td>-0.00254058</td>
      <td>...</td>
      <td>-0.000475807</td>
      <td>-0.0140402</td>
      <td>0.0030798</td>
      <td>0.000438266</td>
      <td>0.00107959</td>
      <td>0.000104271</td>
      <td>-0.000691795</td>
      <td>0.000613057</td>
      <td>-0.000109924</td>
      <td>-0.000521939</td>
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
      <td>-0.00218637</td>
      <td>-0.00100059</td>
      <td>0.00164908</td>
      <td>...</td>
      <td>-0.00290549</td>
      <td>-0.0443946</td>
      <td>0.00347342</td>
      <td>0.000299858</td>
      <td>0.00062553</td>
      <td>0.000165619</td>
      <td>-0.00041004</td>
      <td>6.26682e-05</td>
      <td>-0.000255383</td>
      <td>-0.000434362</td>
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
      <td>-0.00361959</td>
      <td>0.00488799</td>
      <td>0.00168936</td>
      <td>...</td>
      <td>-0.000275847</td>
      <td>-0.00485629</td>
      <td>0.00108332</td>
      <td>-0.000354953</td>
      <td>0.00052248</td>
      <td>0.00164634</td>
      <td>-0.002049</td>
      <td>0.00215299</td>
      <td>-0.00409516</td>
      <td>0.00478023</td>
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
      <td>-0.00169472</td>
      <td>-0.00609196</td>
      <td>0.00286522</td>
      <td>...</td>
      <td>0.00257447</td>
      <td>0.00106701</td>
      <td>0.00191582</td>
      <td>-0.000499389</td>
      <td>0.000659427</td>
      <td>0.000257282</td>
      <td>-0.00259264</td>
      <td>-0.000812146</td>
      <td>-0.000417269</td>
      <td>0.00120451</td>
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
      <td>-0.00250572</td>
      <td>0.0110911</td>
      <td>0.00379146</td>
      <td>...</td>
      <td>0.0169083</td>
      <td>0.0118159</td>
      <td>0.00324122</td>
      <td>1.71778e-05</td>
      <td>0.000502871</td>
      <td>-0.0019061</td>
      <td>-0.000956135</td>
      <td>0.00131178</td>
      <td>0.000335653</td>
      <td>-0.000305509</td>
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
      <td>-0.000676541</td>
      <td>0.00236571</td>
      <td>-0.00542808</td>
      <td>...</td>
      <td>-0.00444412</td>
      <td>0.00799384</td>
      <td>-0.00153584</td>
      <td>-0.00221796</td>
      <td>0.000654807</td>
      <td>0.00175826</td>
      <td>0.00022228</td>
      <td>0.00097945</td>
      <td>-0.00237923</td>
      <td>0.00154693</td>
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
      <td>-0.000206929</td>
      <td>0.00144706</td>
      <td>-0.000719776</td>
      <td>...</td>
      <td>0.00679757</td>
      <td>-0.000338297</td>
      <td>-0.00351584</td>
      <td>0.00180167</td>
      <td>0.000462584</td>
      <td>0.00303921</td>
      <td>0.00434775</td>
      <td>0.00354968</td>
      <td>-2.89882e-05</td>
      <td>0.0117052</td>
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
      <td>0.00192633</td>
      <td>0.00080181</td>
      <td>-0.00546843</td>
      <td>...</td>
      <td>0.00212394</td>
      <td>-0.00339074</td>
      <td>-0.00372017</td>
      <td>-0.000963446</td>
      <td>-1.50815e-05</td>
      <td>0.00101539</td>
      <td>-0.00130499</td>
      <td>0.00107728</td>
      <td>-0.00114766</td>
      <td>-0.000493821</td>
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
      <td>-0.000819385</td>
      <td>8.74538e-05</td>
      <td>0.00113418</td>
      <td>...</td>
      <td>0.000366625</td>
      <td>0.00208452</td>
      <td>-0.00438712</td>
      <td>-0.000486356</td>
      <td>0.00141115</td>
      <td>0.00300085</td>
      <td>-0.000855138</td>
      <td>-0.00527713</td>
      <td>-0.0038038</td>
      <td>-0.00121933</td>
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
      <td>-0.000753905</td>
      <td>-0.000456456</td>
      <td>0.00402603</td>
      <td>...</td>
      <td>0.000891216</td>
      <td>-0.00229696</td>
      <td>-0.00692422</td>
      <td>-0.00129927</td>
      <td>-0.00251864</td>
      <td>0.000189614</td>
      <td>-0.00128221</td>
      <td>-0.00567603</td>
      <td>-0.00219876</td>
      <td>0.000853091</td>
    </tr>
  </tbody>
</table>
<p>1203 rows × 108 columns</p>
</div>



Each _f\_n_ column corresponds to factor of the spectral decomposition of one interaction mode of the interaction matrix of one day.


```python
# Load location data, because we need it to compute the CCA projection
loc_df = bb_network_decomposition.data.load_location_data(supplementary_data_path)
```


```python
cca_factor_df, cca = bb_network_decomposition.projection.get_cca_projection(
    factor_df, loc_df, return_cca=True, num_components=3
)
cca_factor_df.sort_values("date", inplace=True)
```

_cca\_factor\_df_ now contains the network age for all individuals on all dates in the dataset.

The column _network\_age_ contains the first dimension of network age (used throughout most of the paper), 
and the second and third dimensions are stored in the columns _network\_age\_1_ and _network\_age\_2_.


```python
factor_df.to_csv("network_age_cca.csv")
```


```python
# list of variables to use as predictors in task allocation regression tasks
variable_names = [
    ["age"],
    ["age", "network_age"],
    ["network_age"],
    ["network_age", "network_age_1"],
    ["network_age", "network_age_1", "network_age_2"],
]

# list of variables to use as dependent variables in regression tasks
targets = [bb_network_decomposition.constants.supplementary_labels] + list(
    map(lambda l: [l], bb_network_decomposition.constants.supplementary_labels)
)

target_cols = bb_network_decomposition.constants.supplementary_labels
```


```python
# load all required supplementary data
sup_df = bb_network_decomposition.data.load_supplementary_data(
    supplementary_data_path,
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
# remove location data from network age dataframe so that we can safely merge in all
# supplementary data
cca_factor_df = cca_factor_df[
    [c for c in cca_factor_df.columns if c not in location_cols]
]
```


```python
sup_df = bb_network_decomposition.data.merge_location_data(cca_factor_df, sup_df)
```


```python
# regression tasks bootstrap
regression_results = bb_network_decomposition.evaluation.get_bootstrap_results(
    sup_df,
    variable_names,
    targets,
    regression=True,
    use_tqdm=True,
    num_bootstrap_samples=8,
)
```



    
    


These results correspond to section 5 of the manuscript: _Network age predicts an individual's behavior and future role in the colony_


```python
# results of bootstrap analysis, grouped by dependent and independent variables, R^2 scores
regression_results.groupby(["predictors", "target"]).fitted_linear_r2.mean()
```




    predictors                               target                                                
    age                                      circadian_rhythm                                          0.331907
                                             circadian_rhythm,days_left,velocity_day,velocity_night    0.172621
                                             days_left                                                 0.012523
                                             velocity_day                                              0.083986
                                             velocity_night                                            0.286554
    age,network_age                          circadian_rhythm                                          0.403065
                                             circadian_rhythm,days_left,velocity_day,velocity_night    0.210112
                                             days_left                                                 0.015316
                                             velocity_day                                              0.095954
                                             velocity_night                                            0.292630
    network_age                              circadian_rhythm                                          0.387444
                                             circadian_rhythm,days_left,velocity_day,velocity_night    0.199390
                                             days_left                                                 0.010289
                                             velocity_day                                              0.112123
                                             velocity_night                                            0.243061
    network_age,network_age_1                circadian_rhythm                                          0.390725
                                             circadian_rhythm,days_left,velocity_day,velocity_night    0.227511
                                             days_left                                                 0.071366
                                             velocity_day                                              0.112483
                                             velocity_night                                            0.266663
    network_age,network_age_1,network_age_2  circadian_rhythm                                          0.433800
                                             circadian_rhythm,days_left,velocity_day,velocity_night    0.226100
                                             days_left                                                 0.065899
                                             velocity_day                                              0.148080
                                             velocity_night                                            0.260883
    Name: fitted_linear_r2, dtype: float64




```python
# multinomial regression for task allocation task
regression_results = bb_network_decomposition.evaluation.get_bootstrap_results(
    sup_df, variable_names, regression=False, use_tqdm=True, num_bootstrap_samples=8,
)
```



    
    



```python
# results of bootstrap analysis, grouped by dependent and independent variables, R_McF^2 scores
regression_results.groupby(["predictors", "target"]).rho_mcf_linear.mean()
```




predictors                               target                                              
age                                      brood_area_total                                        0.546260
                                         dance_floor                                             0.417913
                                         dance_floor,honey_storage,brood_area_total,near_exit    0.415424
                                         honey_storage                                           0.026411
                                         near_exit                                               0.314711
age,network_age                          brood_area_total                                        0.584782
                                         dance_floor                                             0.512601
                                         dance_floor,honey_storage,brood_area_total,near_exit    0.475106
                                         honey_storage                                           0.051840
                                         near_exit                                               0.376673
network_age                              brood_area_total                                        0.555143
                                         dance_floor                                             0.477920
                                         dance_floor,honey_storage,brood_area_total,near_exit    0.443385
                                         honey_storage                                           0.003832
                                         near_exit                                               0.357418
network_age,network_age_1                brood_area_total                                        0.577571
                                         dance_floor                                             0.477131
                                         dance_floor,honey_storage,brood_area_total,near_exit    0.462893
                                         honey_storage                                           0.166705
                                         near_exit                                               0.386814
network_age,network_age_1,network_age_2  brood_area_total                                        0.575683
                                         dance_floor                                             0.499749
                                         dance_floor,honey_storage,brood_area_total,near_exit    0.475821
                                         honey_storage                                           0.160835
                                         near_exit                                               0.445180
Name: rho_mcf_linear, dtype: float64


These results correspond to section 3 of the manuscript: _Network age correctly identifies task allocation_

## Citation
Social networks predict the life and death of honey bees<br/>
Benjamin Wild, David M Dormagen, Adrian Zachariae, Michael L Smith, Kirsten S Traynor, Dirk Brockmann, Iain D Couzin, Tim Landgraf<br/>
bioRxiv 2020.05.06.076943; doi: https://doi.org/10.1101/2020.05.06.076943 
