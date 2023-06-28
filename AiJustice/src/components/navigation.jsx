import React from "react";
import logo from "../images/logo.png"
import { Link } from "react-router-dom";
import {
  auth,
  registerWithEmailAndPassword,
  logout,
} from "../firebase";
export const Navigation = (props) => {
  return (
    <nav id="menu" className="navbar navbar-default navbar-fixed-top">
      <div className="">
        <div
          className="collapse navbar-collapse"
          id="bs-example-navbar-collapse-1"
        >
          <ul className="nav navbar-nav navbar-left">

            <li>
              <img style={{ width: '170px', height: '150px' }} src={logo} alt="" />{" "}

            </li>
            <li>
              <p className="navbar-brand page-scroll" href="#page-top">
                Ceylon LawMate<span className="navbar-brand-text" style={{}}><br /><br/>Bringing Data into the Sri lankan Courtroom</span>
              </p>
            </li>

          </ul>
          <ul className="nav navbar-nav navbar-right" style={{marginTop:'40px'}}>
            <li>
              <a href="/home" className="page-scroll">
                Home
              </a>
            </li>
            <li>
              <a href="#about" className="page-scroll">
                Services
              </a>
            </li>
            <li>
              <a href="#services" className="page-scroll">
                About
              </a>
            </li>
            <li>
              <Link to="/" className="page-scroll">
               Log in
              </Link>
            </li>
            
            <li>
              <a href="#contact" className="page-scroll">
                Contact
              </a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};
