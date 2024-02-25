# CIFAR-10 Image Classification
 Using some of the deep learning techniques such as CNN (Convolutional Neural Network), InceptionV3, MobileNet and DenseNet and compare their metrics to see which of these will be giving a good functioning model to classify the images.

 # NOTE:-

To run this project you need to have either mini-conda/anaconda environmment installed in your system along with many of the requrired libraries which are mentioned in the module "2.1" in this file and please go thuroug the instructions and documentations properly.

# Chapter 1: Introduction
In the realm of Artificial Intelligence, image processing, prediction, and classification represent greatest
advancements. Initially, AI primarily focused on addressing problems such as finding the shortest paths or
predicting task outcomes based on historical data. However, with the revolutionary progress in image
processing, researchers and computer scientists have delved into exploring diverse applications using these AI
models. A notable application of image processing and classification lies in the field of bioinformatics,
particularly in classifying various types of brain tumours and determining their presence in individuals.
Additionally, these techniques find utility in object identification within satellite images, predominantly for
target identification in military operations. An advanced application involves the discovery and detection of
celestial objects in outer space.

## 1.1 What is Image Classification?

Image classification stands out as one of the most indispensable advancements in modern technology. To
simplify, image classification involves labelling multiple pictures of the same class with their respective names
and passing them through a Machine Learning or Deep Learning model. The goal is to refine this model
iteratively until it becomes self-sufficient at identifying images within that specific class with minimal to no
human intervention, ensuring accurate classification. This application of Artificial Intelligence is highly sought-
after and practical. Subsequent sections will investigate different types of classification.

<img width="958" alt="Fig 1 1" src="https://github.com/JayanthBobbili/CIFAR-10-Image-Classification/assets/97346854/b47c4ca4-2a98-4c4b-83a1-39916c650455">

                     Fig 1.1 Simple representation of image classification using deep learning by SuperAnnote.

## 1.2 Types of Image Classifications

Depending on the data which is given to you and the problem that must be solved there are different types of
image classification processes that can used to solve the problem. The image classification types are as
followed:

1. Binary classification
2. Multiclass Classification
3. Multilabel Classification
4. Hierarchical Classification

### 1.2.1 Binary Classification

As the name implies, Binary Classification categorizes data into two distinct labels or categories. For instance,
it determines whether MRI images of the brain contain a tumor or not, or whether a given dataset of images
classifies as containing animals or not. Essentially, any dataset consisting of images that can be sorted into 'two'
classes falls under the category of Binary Classification.

### 1.2.2 Multiclass Classification

In this type of classification, the dataset given contains the images which could be classified into multiple
classes. For instance, if you were given CIFAR-10 dataset it contains 10 classes, airplane, car, bird, cat, deer,
dog, frog, horse, ship, truck each of these classes has 6000 images with labels they are used to for training a
Neural Network model for multi-class classification.

### 1.2.3 Multilabel Classification

In this type of classification, the dataset given contains the images which has multiple labels to it. This is kind
of an extended version of the Multiclass Classification. For instance, if you were given dataset of Animal in
which an animal contains label like pet/wild/cattle, spices, genes etc… for a single image of an animal. This
type of classification is most useful in bioinformatics and planetary object detection.

### 1.2.4 Hierarchical Classification

In this type of classification, the given dataset of images is classified based upon their hierarchies the higher-
level classes are broader version of the classification and the lower levels of classifications contains more
specific and detailed classification of the images.

# Chapter 2 : Methodology

Using deep learning techniques an Image classification model has been developed with the CIFAR 10 image
data set. Now we will be seeing the steps involved in creation of this model and investigate each step to analyze
for further usage of this model or further development of the model in the future for writing a research paper.
Now let’s look at the attributes of our methodology and investigate them step by step.

1. Modules used.
2. Dataset
3. Data preprocessing and analysis
4. Deep Learning models
5. Evaluation of the models

## 2.1 Modules

We use python programming language for accomplishing this task. Python is an end-to-end programming
language which contains many of modules for performing different tasks while doing projects like Machine
learning and Deep Learning Now let us investigate these modules and investigate their use in this particular
project.
The modules used are as followed.
1. Pandas
2. NumPy
3. Matplotlib and Seaborn
4. Tensorflow

## 2.1.1 Pandas
Pandas is one of the most important modules of python in terms of Machine Learning and Data Science this
model is used to manipulate, analyse and preprocess any kind of data so basically the main function of the
pandas is to tabularize the data in such a way that it would be easy for the user to perform feature engineering
and model training.

## 2.1.2 NumPy
NumPy is basically the ‘mathematical operational’ module of the python in simple terms the NumPy module is
used to convert any kind of data into numerical form and use ML and DL algorithms on them to find the patterns
in them and train the model accordingly to either classify, cluster, predict the outcome of the data. NumPy is
very fast and efficient in calculating or crunching the numbers.

## 2.1.3 Matplotlib and Seaborn
Matplotlib and Seaborn are the representational tools of the python these are mainly used to visualize the data
and see the behavior of the data based on their plotting. Matplotlib is a plotting tool which takes the data and
plots on the graph and has different types of representations where seaborn is used to enhance the visualization
as contain libraries which plots more complex graph plots and their visual representation palettes.

## 2.1.4 TensorFlow
• TensorFlow is an end-to-end open-source platform and python module developed by GOOGLE to
perform many advance steps in single statement of the python program.

• From Keres library of TensorFlow we can create a deep learning model and import some of the
preprocessed datasets.

• We can perform model fitting and model compilations using these modules.

• We can even save models for later purpose and and load them and use them for predictions classifications
and many more.

• This is one of the robust and efficient packages of python that can be used to in Machine Learning and Data Science.

# 2.2 Dataset: CIFAR 10
This dataset is developed by Canadian Institute for Advanced Research, this dataset is used for developing
machine learning algorithms and computer vision techniques. This data consists of 10 classes of images each
class of image of size 32*32*6 and each class of image has exactly 6000 images where 5000 are used for
training and 1000 for testing of the data.

<img width="824" alt="Screenshot 2024-02-24 at 2 56 04 PM" src="https://github.com/JayanthBobbili/CIFAR-10-Image-Classification/assets/97346854/ce4aed5a-d11c-4e9a-9e0f-8b7157a841b4">


## 2.3 Data preprocessing and analysis
Data preprocessing is one of the most important steps of the training a deep learning technique.

• One of the initial steps for preprocessing of the data is to divide data into training and testing datesets.

• Afterwards datasets are normalized by dividing the images with 255.0.

• Thereafter we one-hot-encode the data labels of the data

One hot encoding is process in which the labels are converted into binary form and divide the labels into 10
classes.

* If we do not perform one hot encoding on the labels of the data, whole labels will be classified as one class
and the model performance will be severly affected by this so this one of the most important step in
preprocessing

• Afterwards we visualize the class distribution with labels to avoid any null values.

• One of the steps for INCEPTIONV3 model is to reshape the data set of images to 75*75*3.

## 2.4 Deep Learning Model
For the purpose of this project, I’ve used 4 Deep Learning Models
1. CNN (Convolutional Neural Network)
2. DenseNet
3. Inceptionv3
4. MobileNet

## 2.4.1 CNN (Convolutional Neural Network)
• Convolutional Neural Network is one of the finest Deep learning technique that can be used for any deep
learning model.
• It contains many techniques like
1. Pooling
2. Convolutional and Dense Layer
3. Activation function

• Pooling is the process in which reduction of dimensionality takes place in the inner matrices.

• Convolutional layer is the inner layers of the deep learning model which extract the higher order features
of an image.

• Dense layer contains neurons which attach all the previous layers and to form one big complex layer to
extract the pattern information.

• Activation function is a function which takes weights of the previous layers and checks them with the
threshold value if it is greater than or equal to the threshold value the next layer will be activated, or
weight will be passed to next layer otherwise it will not.

• We use ‘relu’ for inner level layers and we use ‘SoftMax’ for outer most or classification layer.
The other neural network models like DenseNet, MobileNet and InceptionV3 are same as CNN with very
little changes.

Now let us look into the Evaluation of models.

## 2.5 Evaluation of Model’s performance.
We are going to evaluate model’s performance based on the following attributes.
1. Loss
2. Accuracy
3. Precision
4. Recall

   
LOSS: - loss indicates the number of bad predictions by the model the value of the loss should be less than 1
and near to or equal to 0.

ACCURACY: - accuracy indicates the total number of correct predictions overall by the model, the value of
the accuracy should be nearly equal to 1.

PRECISION: - precision measure the number of true positives is correctly predicted over total number of true
positives.

RECALL: - recall measure the total number of relevant datapoints which are taken in consideration while
making a prediction.

# Chapter 3 : Conclusions and Future Works

## Conclusion:
In conclusion, this project successfully implemented image classification using deep learning on the CIFAR-
10 dataset. A Convolutional Neural Network (CNN) was built and optimized, demonstrating competitive
results. Further exploration involved the integration of advanced architectures such as Dense Net, Inception,
and Mobile Net. Training was monitored with informative call-backs like Early Stopping and Model
Checkpoint. The project provided valuable insights into model performance, showcasing the significance of
experimentation, and fine-tuning for optimal results.


The model InceptionV3 is giving highest accuracy of 90.36%.

## Future Work:
For future work, the project could be extended in several directions. Firstly, hyperparameter tuning and
architecture optimization could be further explored to enhance model performance. Additionally, incorporating
more sophisticated data augmentation techniques and exploring transfer learning from larger datasets might
yield improvements. Experimenting with different pre-processing methods or investigating domain-specific
adaptations could also be considered. Lastly, deploying the trained models for real-world applications or
integrating them into a larger system would be an essential step towards practical use. Publishing a research 
paper at conference. The continuous evolution of deep learning techniques offers an exciting avenue for 
ongoing enhancements in image classification projects.







