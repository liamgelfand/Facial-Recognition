import React, { useRef, useEffect } from 'react';

function Webcam() {
    const videoRef = useRef(null);

    useEffect(() => {
        async function setupCamera() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        videoRef.current.play();
                    }
                } catch (error) {
                    console.error('Error accessing the webcam:', error);
                }
            } else {
                console.error('MediaDevices interface not available.');
            }
        }

        setupCamera();
    }, []);

    return (
        <div>
            <video ref={videoRef} style={{ filter: 'grayscale(100%)', width: '640px', height: '480px' }} autoPlay muted />
        </div>
    );
}

export default Webcam;
