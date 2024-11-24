import React, { useState } from 'react';
import config from '../assets/config.json';

const ProcessCCTVPage = () => {
  const backendUrl = config.backendURL;
  const [videoFile, setVideoFile] = useState(null);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resultData, setResultData] = useState([]); 
  const userId = localStorage.getItem('userId');
  const username = localStorage.getItem('username');
  const userEmail = localStorage.getItem('userEmail');

  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!videoFile) {
      setMessage('Please select a video file to upload.');
      return;
    }

    setMessage('');
    setIsLoading(true); 
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('userId', userId);
    formData.append('username', username);
    formData.append('userEmail', userEmail);

    try {
      const response = await fetch(`${backendUrl}/processfootage`, {
        method: 'POST',
        body: formData,
        credentials: 'include', 
      });
      if (!response.ok) {
        throw new Error('Failed to process video.');
      }

      const result = await response.json();
      console.log('Backend Response:', result.data);  
      setMessage(result.status);
      setResultData(result.data);
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);  
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Process CCTV Footage</h1>
      {message && <p className="mb-4 text-center">{message}</p>}

      {isLoading ? (
        <p className="text-blue-500 text-center mb-4">Loading... this will take a long time, please be patient.</p>
      ) : (
        <form onSubmit={handleSubmit} className="flex flex-col items-center">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="mb-4"
          />
          <button
            type="submit"
            className="py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Upload and Process Video
          </button>
        </form>
      )}

      {resultData && (
        <div className="mt-6">
          <h2 className="text-xl">Processing Results:</h2>
          <table className="table-auto w-full mt-4 border-collapse">
            <thead>
              <tr>
                <th className="border p-2">Tracking ID</th>
                <th className="border p-2">Roll No</th>
                <th className="border p-2">Confidence</th>
                <th className="border p-2">Euclidean Distance</th>
                <th className="border p-2">Image Path</th>
              </tr>
            </thead>
            <tbody>
              {resultData.map((s,index) => (
                <tr key={index}>
                  <td className="border p-2">{s.trackingID}</td>
                  <td className="border p-2">{s.rollNo}</td>
                  <td className="border p-2">{s.confidence}</td>
                  <td className="border p-2">{s.euclideanDistance}</td>
                  <td className="border p-2">{s.imagePath}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ProcessCCTVPage;
