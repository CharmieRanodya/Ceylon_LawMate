import React, { useEffect, useState } from "react";
import {
  auth,
  registerWithEmailAndPassword,
  signInWithGoogle,
} from "../firebase";



function About() {

  return (
    <div className="main-body h-screen overflow-hidden">
        <div className="body-container h-screen  text-center relative object-fill" >
        
            <div className="reset_container absolute space-y-10 top-1/2 left-1/2  -translate-y-1/2 -translate-x-1/2 flex flex-col p-20 w-1/2  rounded-lg" >
            <div>
            <h2 className="capitalize text-[#564740] text-[18pt]">About</h2>
            <p>Bringing data into the Sri Lankan courtroom.</p>
            </div>
              
              <p className="text-[#c7ae9b] text-[12pt]">The amount of data that is being recorded across all sectors and domains has been growing, and this tendency is likely to continue. While several sectors already make use of this data to advance their respective professions, the legal sector has even greater potential to benefit from it, specifically for jurisdictional purposes.  </p>

              <p className="text-[#c7ae9b] text-[12pt]">The purpose of this study is to improve the administration of justice in Sri Lanka by providing assistance to key stakeholders in the field through the utilization of data pertaining to the country's various jurisdictions and legal systems. 
</p>
                


            </div>

            

        </div>
      
    </div>
  );
}
export default About;