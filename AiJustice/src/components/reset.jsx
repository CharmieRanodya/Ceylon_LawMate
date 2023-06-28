import React, { useEffect, useState } from "react";
import { useAuthState } from "react-firebase-hooks/auth";
import { Link, useNavigate } from "react-router-dom";
import {
  auth,
  registerWithEmailAndPassword,
  signInWithGoogle,
} from "../firebase";



function Reset() {

  return (
    <div className="main-body h-screen ">
        <div className="body-container h-screen  text-center relative object-fill" >
         



            <div className="reset_container absolute space-y-10 top-1/2 left-1/2  -translate-y-1/2 -translate-x-1/2 flex flex-col p-20 w-1/2  rounded-lg" >

              <h2 className="capitalize text-[#564740] text-[18pt]">Forgot your Password?</h2>
              <p className="text-[#c7ae9b] text-[14pt]">To retrieve your new password, please contact the information Technology Department of Bar Association in Sri Lanka (BASL). </p>


                <div className="contact-info text-center p-5 justify-center ">
                    <div className="contact flex flex-col">
                        <span className="text-[#c7ae9b] text-[12pt]">Telephone : <span className="text-[#564740] "> 0112 345 678</span></span>
                        <span className="text-[#c7ae9b] text-[12pt]">Email : <span className="text-[#564740] "> barassociation@aifj.lk</span></span>
                    </div>
                </div>


            </div>

            

        </div>
      
    </div>
  );
}
export default Reset;