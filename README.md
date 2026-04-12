# Trash Tracker: Iterative Development of a Mobile-Optimized Waste Classifier

## 1. Introduction

Recycling is one of the best ways to help the environment, but it is actually a lot harder than most people think. Many people want to recycle their trash correctly, but they often get confused by all the different rules for materials like glass, plastic, and paper. This leads to a lot of mistakes where people throw the wrong things in the recycling bin, which can actually ruin the whole batch of recycling. Right now, there are not many easy tools that help a normal person know exactly what they are holding and where it should go while they are standing in front of a trash can.

The goal of this project is to build a mobile app that solves this problem using artificial intelligence. I am developing a system that can look at an object through a phone camera and instantly tell the user what category it belongs to. The model is trained to recognize six main types of waste: biodegradable, cardboard, glass, metal, paper, and plastic. By putting this technology on a smartphone, I want to make it easy for anyone to get a fast and accurate answer about their trash.

One of the most important parts of this project is making sure the AI works directly on the phone. I do not want the app to send data to the cloud or need a fast internet connection to work. This means the model runs locally on the device making the app much faster and keeps the user's data private. The main challenge is to keep the model small enough for a phone but smart enough to get the classification right every time.

## 2. System Architecture
The project is split into two main parts: the machine learning model and the mobile application. For the brain of the app, I am using a type of artificial intelligence called YOLO, which stands for You Only Look Once. I chose the Nano version of this model because it is designed to be very lightweight. This is important because a normal phone does not have the same power as a big computer, so the model needs to be small to run smoothly without lagging. To get the model ready, I used a high-performance computer with an NVIDIA RTX 5080 GPU to run the training process. This allowed me to test different settings quickly. Once the training is finished and the model is accurate enough, I will convert it into TFLite format. This format is specifically made for mobile devices and helps the model run efficiently on a phone's processor.

The second part of the project is the mobile app itself, which I am building using Flutter. Flutter is a framework that lets the app run on both Android and iOS. The app will use the phone's camera to see the trash and then use the TFLite model to figure out what it is. Everything happens locally on the phone, so the user just has to point their camera at a piece of trash to see the classification pop up on the screen in real time.

## References
