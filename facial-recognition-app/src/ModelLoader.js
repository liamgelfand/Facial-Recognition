import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

function ModelLoader({ children }) {
    const [model, setModel] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadModel() {
            try {
                const loadedModel = await tf.loadGraphModel('/tfjs_model/model.json');
                setModel(loadedModel);
                console.log('Model loaded successfully');
            } catch (err) {
                console.error('Failed to load the model:', err);
                setError(err);
            }
        }
    
        loadModel();
    }, []);

    return (
        <div>
            {isLoading ? (
                <p>Loading model...</p>
            ) : error ? (
                <div>Error loading model: {error.toString()}</div>
            ) : (
                // Pass the model to children as a prop
                children({ model })
            )}
        </div>
    );
}

export default ModelLoader;
