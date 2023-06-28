import React, { useState, useEffect } from "react";
import { Navigation } from "../components/navigation";
import { Header } from "../components/header";
import { Features } from "../components/features";
import { About } from "../components/about";
import { Testimonials } from "../components/testimonials";
import bg2 from '../images/bg2.jpg'
import JsonData from "../data/data.json";
import SmoothScroll from "smooth-scroll";
import { Link } from "react-router-dom";

import "../App.css";

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
    <div className="main-body h-screen ">
     
      {/* <Navigation /> */}
      {/* <About data={landingPageData.About} /> */}
      {/* <Header data={landingPageData.Header} /> */}
      {/* <Features data={landingPageData.Features} /> */}
      {/* <Testimonials data={landingPageData.Testimonials} /> */}


      <div className="body-container h-screen  text-center relative object-fill bg-cover bg-no-repeat bg-center" style={{ backgroundImage: `url(${bg2})` }}>
      {/* <img src={bg2}  alt="" className="w-full h-full object-cover opacity-80 bg-fixed"/> */}
        <div className="login__container absolute bg-opacity-30 mt-[220px] left-1/2 -translate-x-1/2 flex flex-col w-[90%] space-y-4 rounded-lg" >



          <div className="heading-container flex flex-col lg:flex-row bg-white p-10 justify-center items-center rounded-lg">

              <div className="intro-body text-left leading-[50px] lg:w-2/3 mb-10">
                <h1 className="font-semibold text-[#564740] text-6xl ">Intelligent Support Services</h1>
                <p className="text-gray-700 font-bold">In Which Intelligent Support Services Are We provide?</p>
                <div className="leading-[20px]">
                <p className="text-gray-700 text-[12pt]">Bringing Al to the courtroom in order to make Sri Lanka a more just nation by increasing the availability of justice for the general public by allowing judges and lawyers to handle more cases with comparability reduced effort and time</p>
                </div>
              </div>

              <div className="service-container lg:w-1/3 col-span-2 flex flex-column justify-center p-5 sm:w-full text-[#564740] hover:text-[#564740]">

                    
                      <div className="service p-3 cursor-pointer mx-10 w-full text-center ">
                        <h3>Q&A <br/> Support</h3>
                        <Link to="/support" > 
                        <div className="service-card p-5 border flex justify-center items-center md:mt-11  text-[62px] w-[160px] h-[160px] sm:w-[120px] sm:h-[120px] mx-auto rounded-lg hover:text-[#887161]">
                            <i className="fa fa-group "></i>
                        </div>
                        </Link>
                      </div>
                    
                    
                      <div className="service p-3 mx-10 cursor-pointer w-full text-center">
                        <h3>Content Summarizing <br/> Support</h3>
                        <Link to="/Summarizing" className="page-scroll">
                        <div className="service-card p-5 border flex justify-center items-center md:mt-11 mt-5 text-[62px] w-[160px] h-[160px] sm:w-[120px] sm:h-[120px] mx-auto rounded-lg hover:text-[#887161]">
                          <i className="fa fa-book "></i>
                        </div>
                        </Link>
                      </div>
                    
              </div>

          </div>


          <div className="bottom-container p-5 bg-white rounded-lg">
                <div className="feature-container ">
                  
                <Features data={landingPageData.Features} />

                </div>
          </div>

            


        </div>


      </div>
     
    </div>
  );
};

export default App;
