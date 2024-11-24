import React from 'react';

const HeroPage = () => {
  return (
    <div className="relative bg-gray-100 min-h-screen flex flex-col">

      <nav className="w-full py-4 bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 flex justify-between items-center">
          <div className="text-2xl font-bold text-blue-600">
            Campus Surveillance
          </div>
          <div className="space-x-4">
            <a href="/login" className="px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600">
              Login
            </a>
            <a href="/register" className="px-4 py-2 text-white bg-green-500 rounded hover:bg-green-600">
              Register
            </a>
          </div>
        </div>
      </nav>

     
      <div className="flex-grow flex items-center justify-center bg-gradient-to-r from-blue-500 to-blue-300">
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Analyze CCTV Footage with Ease
          </h1>
          <p className="text-lg md:text-2xl text-white mb-8">
            Manage and identify faces at your campus entrance with our AI-powered system. Ideal for colleges, offices, and security management.
          </p>
          <a
            href="/get-started"
            className="px-6 py-3 text-lg font-semibold text-blue-600 bg-white rounded-lg hover:bg-gray-200">
            Get Started
          </a>
        </div>
      </div>


      <footer className="w-full py-4 bg-white text-center">
        <p className="text-gray-500">© 2024 Campus Surveillance. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default HeroPage;
