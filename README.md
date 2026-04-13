# Trash Tracker: Iterative Development of a Mobile-Optimized Waste Classifier

## 1. Introduction

Recycling is one of the best ways to help the environment, but it is actually a lot harder than most people think. Many people want to recycle their trash correctly, but they often get confused by all the different rules for materials like glass, plastic, and paper. This leads to a lot of mistakes where people throw the wrong things in the recycling bin, which can actually ruin the whole batch of recycling. Right now, there are not many easy tools that help a normal person know exactly what they are holding and where it should go while they are standing in front of a trash can.

The goal of this project is to build a mobile app that solves this problem using artificial intelligence. I am developing a system that can look at an object through a phone camera and instantly tell the user what category it belongs to. The model is trained to recognize six main types of waste: biodegradable, cardboard, glass, metal, paper, and plastic. By putting this technology on a smartphone, I want to make it easy for anyone to get a fast and accurate answer about their trash.

One of the most important parts of this project is making sure the AI works directly on the phone. I do not want the app to send data to the cloud or need a fast internet connection to work. This means the model runs locally on the device making the app much faster and keeps the user's data private. The main challenge is to keep the model small enough for a phone but smart enough to get the classification right every time.

## 2. System Architecture
The project is split into two main parts: the machine learning model and the mobile application. For the brain of the app, I am using a type of artificial intelligence called YOLO, which stands for You Only Look Once. I chose the Nano version of this model because it is designed to be very lightweight. This is important because a normal phone does not have the same power as a big computer, so the model needs to be small to run smoothly without lagging. To get the model ready, I used a high-performance computer with an NVIDIA RTX 5080 GPU to run the training process. This allowed me to test different settings quickly. Once the training is finished and the model is accurate enough, I will convert it into TFLite format. This format is specifically made for mobile devices and helps the model run efficiently on a phone's processor.

The second part of the project is the mobile app itself, which I am building using Flutter. Flutter is a framework that lets the app run on both Android and iOS. The app will use the phone's camera to see the trash and then use the TFLite model to figure out what it is. Everything happens locally on the phone, so the user just has to point their camera at a piece of trash to see the classification pop up on the screen in real time.

## 3 Model Training
### 3.1 Phase 1: Baseline Training and Results
The first step of the project was to run an initial training session to see how well a small model could handle the garbage dataset. I used a dataset of about 10,000 images that were divided into seven categories. For the hardware, I used an NVIDIA RTX 5080 GPU, which made the training much faster. I ran the process for 100 epochs, which took about four hours to complete. This gave me a baseline model that I could use to measure progress in the future.

After the training, the model gave me several metrics to show how it performed. One of them is called mAP50, which stands for mean Average Precision. The model got a score of 0.60 (or 60%). This means that in this first phase, the model is correct about 60% of the time when it identifies an object. For precision and recall for each category, Glass had a high precision of 88%, meaning when the model says something is glass, it is usually right. However, Paper had a very low score of only 15%. This happened because the dataset was not balanced. There were over 13,000 instances of biodegradable waste but only 33 instances of paper. Because of this, the model "biased" its guesses toward the bigger categories since it saw them so much more often during training.

<div align="center">
    <img src="document_images/phase1_metrics.png" width="600">
    <p align="center"><strong>Fig. 1. </strong>Breakdown of precision, recall, and mAP scores across the seven garbage categories.<p>
</div>

To see exactly how the model was thinking, I ran it on some test images. The results were saved in a folder called predict. In this folder, the model takes the original images and draws bounding boxes over what it finds. Each box has a label and a "confidence score" showing how sure the model is. In one test, the model looked at a clear glass bottle but labeled it as PLASTIC 0.80. Because clear glass and clear plastic look very similar, the model got confused. This shows it needs more varied examples of glass in the next phase.

<div align="center">
    <img src="document_images/phase1_glass.jpg" width="300">
    <p align="center"><strong>Fig. 2. </strong>An example of model confusion where a clear glass object was incorrectly identified as plastic with an 80% confidence score.<p>
</div>

On images like the tomatoes and the cardboard box, the model drew way too many boxes. It found the object correctly, but it created extra boxes for different parts of the same item. This makes the screen look messy and confusing for a user.

<div align="center">
    <img src="document_images/phase1_tomatoes.jpg" width="300">
    <p align="center"><strong>Fig. 3. </strong>The model generates multiple overlapping bounding boxes for individual items, such as these tomatoes, creating a cluttered output.<p>
</div>

<div align="center">
    <img src="document_images/phase1_cardboard.jpg" width="300">
    <p align="center"><strong>Fig. 4. </strong>Duplicate "Cardboard" detections on a single box.<p>
</div>

In another case, a cardboard bar box was labeled as PAPER 0.86. This is likely because cardboard and paper have similar textures, and since the model has very few examples of paper, it is struggling to tell the two apart correctly.

<div align="center">
    <img src="document_images/phase1_cardboard2.jpg" width="300">
    <p align="center"><strong>Fig. 5. </strong>A cardboard box incorrectly labeled as "Paper" with 86% confidence, likely due to visual similarities and a small paper dataset.<p>
</div>

Even with the accuracy issues in some categories, the model proved that it is fast enough for a phone. On my computer, it only took about 1.3 milliseconds to process a single image. This is a very good sign because it means that even on a slower phone processor, the app should still be able to show results in real time without any lag.



## References
