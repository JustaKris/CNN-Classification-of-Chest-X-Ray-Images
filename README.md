# Chest X-Ray Image Classification

This project is a continuation of a course project I did for my Deep Learning course in SoftUni. I wanted to adapt my work and build an app around it which would use the best performing model to be able to accept images via user input and generate predictions along with an overlayed model activations heatmap (GRAD_Cam). One of the main goals was to be able to deploy this app to one of the cloud computing platform. 

## Jupyter Notebook

All research, model training and optimization was done in Jupuyter Notebook. The research contains training of various models from scrath as well as transfer learning using a MobileNet model which was chosen due to the limited resources avaialbe to me for model training (AMD RYZEN 7 3700x CPU).

Link to Notebook -> [CNN-classification-of-chest-X-Ray-Images](./notebooks/CNN-classification-of-chest-X-Ray-Images.ipynb)

### Dataset

The dataset was taken from Kaggle and it is comprised of three separate dataets that can also be found on the website. 
It is comprised of <b>7135</b> chest x-ray images. The dat ais split into 3 folders - <b>train/test/val</b> - and each class is stored in its own subfolder - <b>Normal/Pneumonia/Covid-19/Tuberculosis</b>.

The goal is to predict one of the four categories:

- Covid-19
- Normal
- Pneumonia
- Tuberculosis

Kaggle Dataset Link -> [https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)

## Chest X-Ray Image Classification Web App

The second part is building an app using what I've learned from the conducted research. The app is designed to be scalable which is achieved by the use of data, model training and prediction pipelines. I have tried to follow python convention so that the app can be deployed to any remote environment.

### Azure Deployment

The predictor app was deployed on Aruze using a Docker container with a GitHub Workflow set up to update said container when a change is made to the repository and re-deploy the app. Azure was my platform of choice in the end since it provides a bit more computing power for the price and the app was able to run smoothly.

Azure Link -> [https://chest-xray-calssification.azurewebsites.net/](https://chest-xray-calssification.azurewebsites.net/)

<!-- ### Render Deployment

There is also a version of the app which uses Streamlit instead of Flask but I don't find it to be as flexible. There is a deployment workflow available for it which instead deploys that version of the app to Render. -->

### Web App Approach

1. User Input: 
    An image can be provided either by loading one fro mthe user's local drive or by providing a link. Regardless of which option is used, the image is then temporarily saved to be used by the app.

2. Image Transformation: 
    The image is then transformed in order for it to be passed to the trained model. The Tensorflow `imagenet` preprocessing function is used to keep the format consistent with what the model expects.
    
3. Model Training: 
    The transfer learning approach provided the best results and a MobileNetV3 model is what is used in the app. For details on the model training methodology, please refer to the jnotebook.

4. Flask App:
    A Flask app houses the user interface where input is received and passed to the prediction pipeline. The app then displays the resulting outcome. It also displayes the original image with an overlayed heatmap of model activations, signifiyng what the model is "looking" at.