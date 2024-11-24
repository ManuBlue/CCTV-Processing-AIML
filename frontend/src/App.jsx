import { Routes, Route } from "react-router-dom";
import Hero from "./pages/heroPage";
import Login from "./pages/login";
import Register from "./pages/register";
import HomePage from "./pages/homepage";
import ProcessCCTVPage from "./pages/processCCTV";
import SampleImagesPage from "./pages/sampleImages";
import RemoveUsersPage from "./pages/removeUsers";
import MakeModelPage from "./pages/makeModel";
import RealTime from "./pages/realTime";
function App() {
  // const token = localStorage.getItem("token");
  return (
    <Routes>
      <Route path="/" element={<Hero />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/processcctv" element={<ProcessCCTVPage />} />
      <Route path="/addsamples" element={<SampleImagesPage />} />
      <Route path="/removeusers" element={<RemoveUsersPage />} />
      <Route path="/makemodel" element={<MakeModelPage/>} />
      <Route path="/RealTime" element={<RealTime/>} />
    </Routes>
  );
}

export default App;
