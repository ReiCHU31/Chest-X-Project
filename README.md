# The Chest X project
### by Phuong T.M. Chu

This is the project that we finished after 12th week of studying **Machine Learning** from scratch.

<p align="center">
<img width="880" height="450" src="https://s3.amazonaws.com/dsg.files.app.content.prod/gereports/wp-content/uploads/2017/05/01182345/lungs.gif">
</p>

## INTRODUCTION
### 1. Motivation
Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. 

Our vision aims to provide automated chest radiology analysis **comparable to professional practicing radiologists**. In addition, we hope to assist various medical settings such as **improved workflow prioritization** and **clinical decision support**. We have faith that our application will **advance global population health initiatives** through large-scale screening.

### 2. Plan
Main steps:

|Task|Time|Progress|
|:-----------------------------|:------------:|:------------:|
|Data collection |1 days|x|
|Data preprocessing |1 days|x|
|Building Model|5 days|x|
|Build Flask App|1 day|x|
|**Total**|**8 days**||

## MATERIALS AND METHODS
### 1. Datasets
Our [**CheXpert**](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset is provided by Stanford Machine Learning group. The dataset:
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

* Building **CNN model**
The architecture is built by Tensorflow and Transfer Learning techniques.
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

### 4. Model performance summary

Our model has the AUC of **88.57 %** for the train dataset and **88.17 %** for the validation dataset. 

### 5. UI

The app contains 2 pages: 
- **Homepage** including project introduction and section to upload chest X-ray image 

![](https://i.imgur.com/2c3JAUK.png)

![](https://i.imgur.com/RTK0PXQ.jpg)

![](https://i.imgur.com/c27JOGf.png)

![](https://i.imgur.com/clV1J1R.png)

![](https://i.imgur.com/RrCnBS6.png)

- **Result page** for users to received results

![](https://i.imgur.com/lnzIR4y.png)

### 6. SETUP ENVIRONMENT
* In order to run our model on a Flask application locally, you need to clone this repository and create the deep learning model `model.h5` followed the instruction inside the Jupyter notebook, then you save the file inside the folder `model`. After that, yout need to set up the environment by these following commands:

```shell
python3 -m pip install --user pipx
python3 -m pipx ensurepath

pipx install pipenv

# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```
* On the Terminal, use these commands:
```
# enter the environment
pipevn shell
pipenv graph
set FLASK_APP=app.py
set FLASK_ENV=development
export FLASK_DEBUG=1
flask run
```
* If you have error `ModuleNotFoundError: No module named 'tensorflow'` then use
```
pipenv install tensorflow==2.0.0beta-1
```
* If  `* Debug mode: off` then use
```
export FLASK_DEBUG=1
```

* Run the model by 

```shell
pipenv run flask run
```

* If you want to exit `pipenv shell`, use `exit`

## CONCLUSION

We successfully **built a deep neural network model** by implementing **Convolutional Neural Network (CNN)** to automatically interprete X-ray image high AUC **88.17 %**.
In addition, we also **built a Flask application** so user can upload their X-ray images and interpret the results.
