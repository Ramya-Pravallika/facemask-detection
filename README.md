# facemask-detection
This is a machine learning model that is used to detect whether the person is wearing a mask or not. The detailed process of creating a model as follows: 
# importing the libraries and tools
Firstly we are importing the machine learning libraries and tools like pandas, numpy, keras, tensorflow, matplotlib etc. pandas provides high performance,fast,data analysis tools for manipulating numeric data. numpy is used to perform scientific computing. 
# uploading the dataset
Using uploaded=files.upload() we can upload our dataset into the notebook
# unzipping the dataset
Initially our dataset is in zip format so we have to unzip the dataset using !unzip data.zip ; !rm data.zip
# Training the dataset
Training the dataset using Imagedatagenerator and train data generator with batch size of 9 and epoch of 35. Plotting some random images from trained dataset representing binary value 0 if person wearing mask otherwise 1
![image](https://user-images.githubusercontent.com/107994772/188714361-5a241edb-440f-42ca-93d3-1496a45fd629.png)
# sequential modelling using convolutional 2D layers and maxpool layers with same padding and activation as relu
Sequencial model allows to create model layer by layer. for sequencial modelling we need residual connection. This modelling can be done by using Convolutional 2D layers and maxpool layers because these layers can do sharing and they had multiple inputs and outputs.
![image](https://user-images.githubusercontent.com/107994772/188714495-a95f1636-37c7-4485-b135-49ee919265d0.png)

Finding accuracy and precision values of trained model with an epoch of 35. Plotting training vs Validation loss and also training vs validation accuracy

![image](https://user-images.githubusercontent.com/107994772/188714082-45001373-10ba-4ae6-9020-d6619b9f6364.png)

# OUTPUT
finally an output image is displayed stating that the person is wearing a mask or not
