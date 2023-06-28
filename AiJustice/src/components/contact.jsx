import React, { useEffect, useState } from "react";
import {
  auth,
  registerWithEmailAndPassword,
  signInWithGoogle,
} from "../firebase";
import charmie from '../images/charmie.jpeg';
import ruchira from '../images/ruchira.jpeg';
import anuki from '../images/anuki.jpeg';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faFacebook, faGithubSquare, faLinkedin, faTwitterSquare } from '@fortawesome/free-brands-svg-icons'
function Contact() {

  return (
    <div className="main-body h-screen overflow-hidden">
        <div className="body-container h-screen  text-center relative object-fill" >
        
            <div className="absolute space-y-10 left-1/2 mt-[220px] -translate-x-1/2 flex flex-col p-20 w-2/3 md:w-[90%]  rounded-lg" >
            
              
             <div className="contact-body p-5 w-full flex mx-auto justify-center space-x-5">

                <div className="card border w-[250px] border-[#564740] border-opacity-20 rounded-md relative h-[280px] shadow-md shadow-slate-200">
                          <div className="card-head p-4 bg-[#564740] h-[100px]"></div>
                          <div className="card-img p2 h-[100px] w-[100px] object-fill bg-slate-200 rounded-full mx-auto absolute left-1/2 -translate-x-1/2 top-[50px]">
                                <img src={anuki} alt="" srcset="" className="w-full h-full object-cover rounded-full"/>
                          </div>
                          <div className="profile mt-[80px]">
                            <span className="text-gray-800  text-[12pt]"> De Silva E.A.A </span>
                            <div className="scoial flex w-full text-[22pt] justify-center space-x-5 mt-5 text-[#564740] ">
                              <a href="https://github.com/ANUKI-ISE"><FontAwesomeIcon icon={faGithubSquare} className="hover:text-[#887161]"/></a>
                              <a href="https://www.linkedin.com/in/anuki-de-silva-08b537209/"> <FontAwesomeIcon icon={faLinkedin} className="hover:text-[#887161]"/></a>
                              <a href="https://twitter.com/anuki_de_silva"><FontAwesomeIcon icon={faTwitterSquare} /></a>
                              <a href="https://www.facebook.com/bonobonoyaa.kookie"><FontAwesomeIcon icon={faFacebook} /></a>

                            </div>
                          </div>
                </div>

                <div className="card border w-[250px] border-[#564740] border-opacity-20 rounded-md relative h-[280px] shadow-md shadow-slate-200">
                          <div className="card-head p-4 bg-[#564740] h-[100px]"></div>
                          <div className="card-img p2 h-[100px] w-[100px] object-fill bg-slate-200 rounded-full mx-auto absolute left-1/2 -translate-x-1/2 top-[50px]">
                                <img src={ruchira} alt="" srcset="" className="w-full h-full object-cover rounded-full"/>
                          </div>
                          <div className="profile mt-[80px]">
                            <span className="text-gray-800  text-[12pt]"> Ruchira Bandara </span>
                            <div className="scoial flex w-full text-[22pt] justify-center space-x-5 mt-5 text-[#564740] ">
                              <a href="#"><FontAwesomeIcon icon={faGithubSquare} className="hover:text-[#887161]"/></a>
                              <a href="https://www.linkedin.com/in/ruchirabandara/"> <FontAwesomeIcon icon={faLinkedin} className="hover:text-[#887161]"/></a>
                            </div>
                          </div>
                </div>

                <div className="card border w-[250px] border-[#564740] border-opacity-20 rounded-md relative h-[280px] shadow-md shadow-slate-200">
                          <div className="card-head p-4 bg-[#564740] h-[100px]"></div>
                          <div className="card-img p2 h-[100px] w-[100px] object-fill bg-slate-200 rounded-full mx-auto absolute left-1/2 -translate-x-1/2 top-[50px]">
                                <img src={charmie} alt="" srcset="" className="w-full h-full object-cover rounded-full"/>
                          </div>
                          <div className="profile mt-[80px]">
                            <span className="text-gray-800  text-[12pt]"> Charmie Ranodya </span>
                            <div className="scoial flex w-full text-[22pt] justify-center space-x-5 mt-5 text-[#564740] ">
                              <a href="https://github.com/CharmieRanodya"><FontAwesomeIcon icon={faGithubSquare} className="hover:text-[#887161]"/></a>
                              <a href="https://www.linkedin.com/in/charmie-ranodya-9794641a7/"> <FontAwesomeIcon icon={faLinkedin} className="hover:text-[#887161]"/></a>
                              <a href="https://twitter.com/Charmie_Ranodya"><FontAwesomeIcon icon={faTwitterSquare} /></a>
                              <a href="https://web.facebook.com/profile.php?id=100083949432068"><FontAwesomeIcon icon={faFacebook} /></a>
                            </div>
                          </div>
                </div>


                

                
                
             </div>
                


            </div>

            

        </div>
      
    </div>
  );
}
export default Contact;