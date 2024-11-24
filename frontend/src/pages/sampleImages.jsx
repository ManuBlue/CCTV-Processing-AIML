import React, { useState } from 'react';
import config from '../assets/config.json';

const SampleImagesPage = () => {
  const backendUrl = config.backendURL;
  const [samples, setSamples] = useState([{ name: '', images: [] }]);
  const [message, setMessage] = useState('');


  const userId = localStorage.getItem('userId');
  const username = localStorage.getItem('username');
  const userEmail = localStorage.getItem('userEmail');

  const handleNameChange = (index, event) => {
    const updatedSamples = [...samples];
    updatedSamples[index].name = event.target.value;
    setSamples(updatedSamples);
  };

  const handleImageChange = (index, event) => {
    const updatedSamples = [...samples];
    updatedSamples[index].images = Array.from(event.target.files);
    setSamples(updatedSamples);
  };

  const addSample = () => {
    setSamples([...samples, { name: '', images: [] }]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setMessage(''); 
    const formData = new FormData();
    formData.append('userId', userId);
    formData.append('username', username);
    formData.append('userEmail', userEmail);

    samples.forEach((sample, i) => {
      formData.append(`names`, sample.name); 
      sample.images.forEach((image) => {
        formData.append(`images`, image); 
      });
    });
    

    try {
      const response = await fetch(`${backendUrl}/addsamples`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload samples.');
      }

      setMessage('Samples uploaded successfully.');
    } catch (error) {
      setMessage(error.message);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Add User Samples to Database</h1>
      {message && <p className="mb-4 text-center">{message}</p>}
      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        {samples.map((sample, index) => (
          <div key={index} className="mb-4 w-full max-w-sm">
            <label className="block text-gray-700 mb-2">Name:</label>
            <input
              type="text"
              value={sample.name}
              onChange={(e) => handleNameChange(index, e)}
              required
              className="border border-gray-300 rounded w-full py-2 px-3 mb-2"
            />

            <label className="block text-gray-700 mb-2">Upload Images:</label>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => handleImageChange(index, e)}
              className="border border-gray-300 rounded w-full py-2 px-3 mb-2"
            />
          </div>
        ))}
        <button
          type="button"
          onClick={addSample}
          className="py-2 px-4 bg-gray-500 text-white rounded hover:bg-gray-600 mb-4"
        >
          Add Another Person
        </button>
        <button
          type="submit"
          className="py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Submit Samples
        </button>
      </form>
    </div>
  );
};

export default SampleImagesPage;
