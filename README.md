# The Chest X project
### by Phuong T.M. Chu

<p align="center">
<img width="660" height="500" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Projectional_rendering_of_CT_scan_of_thorax_%28thumbnail%29.gif/358px-Projectional_rendering_of_CT_scan_of_thorax_%28thumbnail%29.gif">
</p>

## INTRODUCTION
### 1. User stories
Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. 

This app 

- aims to provide **automated chest radiograph interpretation** at the **level of practicing radiologists** 

- could provide substantial benefit in many medical settings, from 
    - **improved workflow prioritization** and **clinical decision** 
    - support to **large-scale screening** and **global population health initiatives**.

### 2. Plan
Main steps:

|Task|Time|Progress|
|:-----------------------------|:------------:|:------------:|
|Data collection |1 days|x|
|Data preprocessing |1 days|x|
|Building Model|10 days|x|
|Build Flask App|1 day|x|
|**Total**|**13 days**||

## MATERIALS AND METHODS
### 1. Datasets
For progress in both development and validation of automated algorithms, we realized there was a need for a labeled dataset that 
- (1) was large, 
- (2) had strong reference standards, and 
- (3) provided expert human performance metrics for comparison.

[**CheXpert**](https://stanfordmlgroup.github.io/competitions/chexpert/)
- contains 224,316 chest radiographs of 65,240 patients
- is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, 
- which features uncertainty labels and radiologist-labeled reference standard evaluation sets.

**Labels**

Each report was labeled for the presence of 14 observations as positive, negative, or uncertain.
<p align="center">
<img width="560" height="350" src="https://stanfordmlgroup.github.io/competitions/chexpert/img/table1.png">
</p>

### 2. Methods
* **Python** and some neccessary libraries such **pandas, numpy, keras, tensorflow, flask**.
* **Google Cloud Platform** to deploy the app and SQL data storage.

### 3. Building Models
* Building **CNN model**

- The training labels in the dataset for each observation are either **0** (negative), **1** (positive), or **U** (uncertain). Explore different approaches to using the uncertainty labels during the model training.

    * **U-Ignore**: ignore the uncertain labels during training.
    * **U-Zeroes**: map all instances of the uncertain label to 0.
    * **U-Ones**: map all instances of the uncertain label to 1.
    * **U-SelfTrained**: first train a model using the U-Ignore approach to convergence, and then use the model to make predictions that re-label each of the uncertainty labels with the probability prediction outputted by the model.
    * **U-MultiClass**: treat the uncertainty label as its own class.

In case of **multi-label image classification**, we can have more than one label for a single image. We want the **probabilities** to be **independent** of each other. Using the softmax activation function will not be appropriate. Instead, we can use the **sigmoid activation** function. This will **predict the probability** for **each class independently**. It will **internally create n models** (n here is the total number of classes), one for each class and predict the probability for each class.

<p align="center">
<img width="660" height="600" src="https://stanfordmlgroup.github.io/competitions/chexpert/img/figure1.png">
</p>

```python
densenet = tf.keras.applications.densenet.DenseNet121(weights='imagenet',input_shape=(224,224,3),include_top=False)
densenet.trainable=False
                                                  
model = tf.keras.Sequential([
    densenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(14, activation = 'sigmoid')])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=["accuracy"])
```

### 4. Model summary

### 5. UI

The app contains 2 pages: 
- **Homepage** including project introduction and section to upload chest X-ray image 
![](https://i.imgur.com/2c3JAUK.png)

![](https://i.imgur.com/RTK0PXQ.jpg)

![](https://i.imgur.com/c27JOGf.png)

![](https://i.imgur.com/clV1J1R.png)

![](https://i.imgur.com/RrCnBS6.png)

- **Result page** for users to received results
![](https://i.imgur.com/QhnMp2J.png)

## CONCLUSION

