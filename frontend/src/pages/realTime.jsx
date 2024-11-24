import React, { useEffect, useRef, useState } from 'react';

const RealTime = () => {
    const [imageSrc, setImageSrc] = useState(null);
    const [connected, setConnected] = useState(false);
    const [loading, setLoading] = useState(false);
    const wsRef = useRef(null);

    const openWebSocket = () => {
        setLoading(true); // Set loading to true when WebSocket is being opened
        wsRef.current = new WebSocket("ws://localhost:8000/ws/video");
        wsRef.current.onopen = () => {
            wsRef.current.send(localStorage.getItem('userId'));
            setConnected(true);
            setLoading(false); // Stop loading once WebSocket is open
        };
        wsRef.current.onmessage = (event) => {
            setImageSrc(`data:image/jpeg;base64,${event.data}`);
        };
        wsRef.current.onclose = () => {
            setConnected(false);
            console.log("WebSocket closed.");
        };
    };

    const closeWebSocket = () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setConnected(false);
        setImageSrc(null);
    };

    const toggleConnection = () => {
        connected ? closeWebSocket() : openWebSocket();
    };

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    return (
        <div style={{ textAlign: 'center', margin: '20px' }}>
            <button
                onClick={toggleConnection}
                style={{
                    padding: '10px 20px',
                    backgroundColor: connected ? '#f44336' : '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    fontSize: '16px',
                    marginBottom: '20px'
                }}
            >
                {connected ? 'Close WebSocket' : 'Open WebSocket'}
            </button>

            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center' }}>
                {loading ? (
                    <p>Loading video stream...</p>
                ) : (
                    imageSrc && <img src={imageSrc} alt="Video stream" style={{ width: '100%', maxWidth: '600px', borderRadius: '10px' }} />
                )}
            </div>
        </div>
    );
};

export default RealTime;
