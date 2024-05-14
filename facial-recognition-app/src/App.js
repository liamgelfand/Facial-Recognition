import React from 'react';
import './App.css';
import Webcam from './Webcam';
import EmotionRecognition from './ModelLoader'; 


function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Webcam></Webcam>
        <EmotionRecognition></EmotionRecognition>
      </header>
    </div>
  );
}

export default App;
