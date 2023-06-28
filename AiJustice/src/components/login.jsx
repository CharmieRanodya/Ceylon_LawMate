import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { auth, logInWithEmailAndPassword, signInWithGoogle } from "../firebase";
import { useAuthState } from "react-firebase-hooks/auth";
import bg from '../images/bg.jpg'

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, loading, error] = useAuthState(auth);
  const navigate = useNavigate();
  useEffect(() => {
    if (loading) {
      // maybe trigger a loading screen
      return;
    }
    // if (user) navigate("/home");
  }, [user, loading]);
  const userLogin=async(email, password)=>{
    console.log('email, password',email, password)
    // await logInWithEmailAndPassword(email, password)
    if(await logInWithEmailAndPassword(email, password)){
      navigate("/home")
    }
  

  }
  return (
    <div className="main-body h-screen ">
        <div className="body-container h-screen  text-center relative object-fill" >
          <img src={bg}  alt="" className="w-full h-full object-cover"/>{" "}



            <div className="login__container absolute bg-opacity-30 border backdrop-blur-sm bg-white top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col p-20 w-[500px] space-y-4 rounded-lg" >

              <h3 className="text-[32pt] text-[#564740]">WELCOME</h3>
              <input
                type="text"
                className="login__textBox p-5 leading-5 rounded-lg"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="E-mail Address"
              />
              <input
                type="password"
                className="login__textBox p-5 rounded-lg"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
              />
              <button
                className="login__btn p-5 bg-gradient-to-r from-[#564740] to-[#887161] rounded-lg text-white text-2xl uppercase font-semibold" 
                onClick={() => userLogin(email, password)}
              >
                Login
              </button>
              <button className="login__btn login__google p-5 bg-blue-500  hover:bg-blue-600 rounded-lg text-white text-2xl  " onClick={signInWithGoogle}>
                Login with Google
              </button>
              <div>
                <Link to="/reset" className="text-blue-500">Forgot Password</Link>
              </div>
              <div className="text-white">
                Don't have an account? <Link to="/register" className="text-blue-500">Register</Link> now.
              </div>
            </div>

        </div>
      
    </div>
  );
}
export default Login;