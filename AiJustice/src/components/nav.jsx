import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import logowhite from '../images/logo.png';

function Navbar() {
  const location = useLocation();
  const [navLinks] = useState([
    { title: "Home", path: "/home" },
    { title: "About", path: "/about" },
    { title: "Contact", path: "/contact" },
    { title: "Log in", path: "/" },
  ]);

  return (
    <nav className="flex justify-between items-center bg-white p-3 fixed z-10 w-full bg-opacity-80">
      <div className="nav-row flex flex-col justify-between mx-auto w-full">
        <div className="nav-body w-[90%] mx-auto flex justify-between">
          <div className="flex items-center">
            <a href="#" className="nav_brand ">
              {" "}
              <img src={logowhite} alt="logo" className="w-[180px]" />{" "}
            </a>
            <div className="ml-5">
              <h1 className="text-[#564740] text-4xl font-bold">Ceylon LawMate</h1>
              <p className="text-[#c7ae9b] text-2xl">Bringing Data into the Sri Lankan Courtroom</p>
            </div>
          </div>

          <div className="flex items-center w-1/3 justify-around  ">
            <ul className="flex mr-5 text-[12pt] text-[#564740] space-x-8">
              {navLinks.map((link, index) => (
                <li key={index} className={`nav_item mr-5 ${link.path === location.pathname ? 'border-b-2 border-[#887161]' : ''}`}>
                  <Link to={link.path} className="nav_link">{link.title}</Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="row felx flex-row columns-6 m-0">
          <div className=" bg-[#f0efeb] py-5"> </div>
          <div className=" bg-[#bebebe] py-5"> </div>
          <div className=" bg-[#887161] py-5"> </div>
          <div className=" bg-[#5b4f45] py-5"> </div>
          <div className=" bg-[#aa9c93] py-5"> </div>
          <div className=" bg-[#867974] py-5"> </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
