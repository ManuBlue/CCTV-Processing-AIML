import React, { useState } from 'react';
import config from '../assets/config.json';

const RemoveUsersPage = () => {
  const backendUrl = config.backendURL;
  const [names, setNames] = useState(['']);
  const [message, setMessage] = useState('');

  const userId = localStorage.getItem('userId');
  const username = localStorage.getItem('username');
  const userEmail = localStorage.getItem('userEmail');

  const handleNameChange = (index, event) => {
    const updatedNames = [...names];
    updatedNames[index] = event.target.value;
    setNames(updatedNames);
  };

  const addNameField = () => {
    setNames([...names, '']);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(''); 
    const payload = {
      userId,
      username,
      userEmail,
      names,  
    };

    try {
      const response = await fetch(`${backendUrl}/removeusers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to remove users.');
      }

      setMessage('Users removed successfully.');
    } catch (error) {
      setMessage(error.message);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Remove Users from Database</h1>
      {message && <p className="mb-4 text-center">{message}</p>}
      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        {names.map((name, index) => (
          <div key={index} className="mb-4 w-full max-w-sm">
            <label className="block text-gray-700 mb-2">Name:</label>
            <input
              type="text"
              value={name}
              onChange={(e) => handleNameChange(index, e)}
              required
              className="border border-gray-300 rounded w-full py-2 px-3 mb-2"
            />
          </div>
        ))}
        <button
          type="button"
          onClick={addNameField}
          className="py-2 px-4 bg-gray-500 text-white rounded hover:bg-gray-600 mb-4"
        >
          Add Another Name
        </button>
        <button
          type="submit"
          className="py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Submit Names to Remove
        </button>
      </form>
    </div>
  );
};

export default RemoveUsersPage;
