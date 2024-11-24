import React, { useState } from 'react';  // Import useState
import { useNavigate } from 'react-router-dom';
import config from '../assets/config.json';

const HomePage = () => {
  const backendUrl = config.backendURL;
  const navigate = useNavigate();
  const username = localStorage.getItem('username');


  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleLogout = () => {
    localStorage.clear(); 
    navigate('/');
  };


  

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Welcome, {username}!</h1>
      <p className="mb-8">What would you like to do?</p>
      <div className="flex flex-col gap-4">
        <button
          onClick={() => navigate('/processcctv')}
          className="py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Process CCTV Footage
        </button>
        <button
          onClick={() => navigate('/addsamples')}
          className="py-2 px-4 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Add User Samples to Database
        </button>
        <button
          onClick={() => navigate('/removeusers')}
          className="py-2 px-4 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Remove Users from Database
        </button>
        <button
          onClick={() => navigate('/makemodel')} // Renamed route
          className="py-2 px-4 bg-purple-500 text-white rounded hover:bg-purple-600"
        >
          Make Model
        </button>
        <button
          onClick={() => navigate('/realTime')}
          className="py-2 px-4 bg-yellow-500 text-white rounded hover:bg-yellow-600"
        >
          Track Real-Time
        </button>
      </div>
      {loading && <p>Loading...</p>} {/* Show loading indicator */}
      {message && <p>{message}</p>} {/* Show the message */}
      <button
        onClick={handleLogout}
        className="mt-8 py-2 px-4 bg-gray-400 text-white rounded hover:bg-gray-500"
      >
        Logout
      </button>
    </div>
  );
};

export default HomePage;
