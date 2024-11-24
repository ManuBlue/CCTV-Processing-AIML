import React, { useState } from 'react';
import config from '../assets/config.json';

const MakeModelPage = () => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const userId = localStorage.getItem('userId');
  const username = localStorage.getItem('username');
  const userEmail = localStorage.getItem('userEmail');
  const backendUrl = config.backendURL;

  const handleMakeModel = async () => {
    setLoading(true);
    setMessage(''); 
    const formData = new FormData();
    formData.append('userId', userId);
    formData.append('username', username);
    formData.append('userEmail', userEmail);

    try {
      const response = await fetch(`${backendUrl}/makemodel`, {
        method: 'POST',
        body: formData, 
      });

      if (!response.ok) {
        throw new Error('Failed to create the model.');
      }

      const result = await response.json();
      setMessage('Model created successfully!');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Make Model</h1>
      {message && <p className="mb-4 text-center">{message}</p>}
      <button
        onClick={handleMakeModel}
        className={`py-2 px-4 ${loading ? 'bg-gray-400' : 'bg-blue-500'} text-white rounded hover:bg-blue-600`}
        disabled={loading}
      >
        {loading ? 'Creating...' : 'Make Model'}
      </button>
    </div>
  );
};

export default MakeModelPage;
