import React, { useState, useEffect  } from "react";
import JsonData from "./data/data.json";
import SmoothScroll from "smooth-scroll";
import { Route, Routes,  BrowserRouter } from "react-router-dom";
import Login from "./components/login";
import Home from "./components/home";
import {Testimonials} from "./components/testimonials";
import {Header} from "./components/header";
import Register from "./components/register";
import Navbar from "./components/nav";
import Reset from "./components/reset";
import About from "./components/about"; 
import Contact from "./components/contact";

import "./App.css";

export const scroll = new SmoothScroll('a[href*="#"]', {
  speed: 1000,
  speedAsDuration: true,
});

const App = () => {
  const [landingPageData, setLandingPageData] = useState({});
  useEffect(() => {
    setLandingPageData(JsonData);
  }, []);



  return (

    <div>

      

      <BrowserRouter>
          <Navbar />
        <Routes>
          <Route exact path="/" element={<Login />} />
          <Route exact path="/home" element={<Home />} />
          <Route exact path="/register" element={<Register />} />
          <Route exact path="/Reset" element={<Reset />} />
          <Route exact path="/About" element={<About />} />
          <Route exact path="/Contact" element={<Contact />} />
          <Route exact path="/Header" element={<Header />} />
          <Route exact path="/summarizing" element={<Header  data={landingPageData.Testimonials} />} />
          <Route exact path="/support" element={<Testimonials  data={landingPageData.Testimonials} />} />
        </Routes>
      </BrowserRouter>
      
    </div>

      
      
  );
};

export default App;
