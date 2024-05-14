import React, { useEffect, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

const EmotionRecognition = () => {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Define the emotion mapping
  const emotionMapping = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/tfjs_model/model.json');
        setModel(loadedModel);
      } catch (error) {
        console.error('Failed to load model:', error);
      }
    };

    const getCameraFeed = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
        };
      } catch (error) {
        console.error('Failed to access camera:', error);
      }
    };

    loadModel();
    getCameraFeed();
  }, []);

  const predictEmotion = async () => {
    if (model && videoRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      context.drawImage(videoRef.current, 0, 0, 48, 48);

      const imageData = context.getImageData(0, 0, 48, 48);
      const tensor = tf.browser.fromPixels(imageData)
        .mean(2) // Convert to grayscale by averaging the color channels
        .expandDims(2) // Add a channel dimension
        .expandDims() // Add a batch dimension
        .toFloat();

      const predictions = model.predict(tensor);
      const predictedClass = predictions.argMax(1).dataSync()[0];
      const predictedEmotion = emotionMapping[predictedClass];
      setPrediction(predictedEmotion);
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      predictEmotion();
    }, 100); // Adjust the interval as needed for your application
    return () => clearInterval(interval);
  }, [model]);

  return (
    <div>
      <h1>Emotion Recognition</h1>
      <video ref={videoRef} width="640" height="480" style={{ display: 'none' }} />
      <canvas ref={canvasRef} width="48" height="48" style={{ display: 'none' }} />
      {prediction !== null && <p>Predicted Emotion: {prediction}</p>}
    </div>
  );
};

export default EmotionRecognition;
