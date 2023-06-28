import React, { useEffect, useState } from "react";
import { useAuthState } from "react-firebase-hooks/auth";
import { Link, useNavigate } from "react-router-dom";
import {
  auth,
  registerWithEmailAndPassword,
  signInWithGoogle,
} from "../firebase";
// import "../components/register.css";
import bg from '../images/bg.jpg'

function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [user, loading, error] = useAuthState(auth);
  // const history = useHistory();
  const navigate = useNavigate();
  const register = async() => {
    if (!name) alert("Please enter name");
    if(await registerWithEmailAndPassword(name, email, password)){
      navigate("/home")
    }
  };
  useEffect(() => {
    if (loading) return;
    // if (user) navigate('/home');;
  }, [user, loading]);


  return (
    <div className="main-body h-screen ">
        <div className="body-container h-screen  text-center relative object-fill" >
          <img src={bg}  alt="" className="w-full h-full object-cover"/>{" "}



            <div className="register__container absolute bg-opacity-30 border backdrop-blur-sm bg-white top-1/2 left-1/2  -translate-y-1/2 -translate-x-1/2 flex flex-col p-20 w-[500px] space-y-4 rounded-lg" >

              
              <input
                type="text"
                className="register__textBox p-5 leading-5 rounded-lg"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Full Name"
              />
              <input
                type="text"
                className="register__textBox p-5 rounded-lg"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="E-mail Address"
              />
              <input
                type="password"
                className="register__textBox p-5 rounded-lg"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
              />


              <button
                className="register__btn p-5 bg-gradient-to-r from-[#564740] to-[#887161] rounded-lg text-white text-2xl uppercase font-semibold" 
                onClick={register}
              >
                Register
              </button>
              <button className="register__btn login__google p-5 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-2xl  " onClick={signInWithGoogle}>
                Register with Google
              </button>
              
              <div className="text-white">
              Already have an account?<Link to="/" className="text-blue-500"> Login</Link> now.
              </div>


            </div>

        </div>
      
    </div>
  );
}
export default Register;