import React ,{CSSProperties}from "react";
import axios from 'axios';
import BarLoader from "react-spinners/ClipLoader";
import { Navigation } from "../components/navigation";

import image from '../images/image.png'

const options = ["facts of the case", "judicial reasoning"];
const override= {
  display: "block",
  margin: "0 auto",
  borderColor: "red",
};

export const Header = (props) => {
  const [selectedFile, setSelectedFile] = React.useState(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [selected, setSelected] = React.useState(options[0]);
  const [response, setResponse] = React.useState({
    "violationData": {
        "Court": "the Supreme Court",
        "DocumentShouldBring": "medical evidence Documents",
        "Lawyers": "Shantha Jayawardena with Niranjan Arulpragasam , Upul Kumarapperuma , Ms. Nayomi Wickramasekera",
        "Suggetion": "Considering all these things, we hold that the Petitioners have not presented their  case to the satisfaction of this Court. We therefore can’t rely on the complaint of both  Petitioners. For the above reasons, we dismiss the Petition of the Petitioner."
    },
    "violationType": "article 11. of the constitution"
});
  const handleSubmit = async(event) => {
    event.preventDefault()
    setIsLoading(false)
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const resp = await axios({
        method: "post",
        url: "http://ec2-13-229-183-94.ap-southeast-1.compute.amazonaws.com:5006/summary",
        data: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log("🚀 ~ file: header.jsx:16 ~ handleSubmit ~ response:", resp)
      setResponse(resp)
      setIsLoading(true)
    } catch(error) {
      console.log(error)
    }
  }
  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0])
  }




  return (
    <div className="main-body h-screen ">
      <div className="body-container h-screen  text-center relative object-fill" >
        
        <div className="summery__container absolute bg-opacity-30 backdrop-blur-sm bg-white top-1/2 left-1/2  -translate-y-1/2 -translate-x-1/2 flex flex-col p-20 w-3/4 space-y-5 rounded-lg" >
              
            <div className="mb-10">
              <h1 className="text-6xl text-[#564740] font-normal ">              
                Case Summarizing Support For A Better Decision 
                <span></span>
              </h1>
              {/* <p className="text-3xl mt-5">To Get Started, Upload A Case File</p> */}
            </div>

          <div className="columns-3 mx-auto justify-between  mt-5 mb-10 flex  items-center text-center  w-full">
              
              
              <div className="flex items-center justify-center mt-5">
                <input type="file" onChange={handleFileSelect} className="bg-gray-200 px-4 py-2 rounded-lg border-2 border-gray-300 focus:outline-none focus:border-blue-500" />
              </div>

              <div className="flex items-center justify-center mt-5">
                <select value={selected} onChange={e => setSelected(e.target.value)} className="bg-gray-200 px-4 py-2 rounded-lg border-2 border-gray-300 focus:outline-none focus:border-blue-500">
                  {options.map((value) => (
                    <option value={value} key={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>

              

                <div>
                  <button
                    type="submit"
                    form="myForm"
                    className="btn w-[120px] text-center mx-auto rounded-full bg-gradient-to-r from-[#564740] to-[#887161] text-white p-3 text-[12pt] uppercase  cursor-pointer hover:text-white hover:from-blue-700 hover:to-blue-500 transition duration-500 delay-1000 ease-in-out ml-[20px]"
                    alt="submit Checkout"
                    
                    onClick={handleSubmit}
                  >
                    submit 
                  </button>
                </div>

                
          </div>
              
             


                <BarLoader loading={!isLoading} height={1} width={1} color="#36d7b7" />
          
              
                {response&&<div style={{ backgroundColor: "#F6ECE8" }}>
                      <div className="testimonial w-full">
                          <div className="testimonial-content w-full">
                              
                          <div className="w-full text">
                          <textarea className="w-[80%] h-[300px] resize-none" />
                          </div>
                    

                {/* <div className="testimonial-meta" style={{marginTop:'20px',color:'black'}}>Court: {response?.violationData?.Court}</div>
                  <p style={{marginTop:'10px',color:'black'}}> DocumentShouldBring: {response?.violationData?.DocumentShouldBring}</p>
                  <p style={{marginTop:'10px',color:'black'}}> Lawyers: {response?.violationData?.Lawyers}</p>
                  <p style={{marginTop:'10px',color:'black'}}> Suggetion: {response?.violationData?.Suggetion}</p>
                  <p style={{marginTop:'10px',color:'black'}}> violationType: {response?.violationType}</p> */}
                
                
                </div>


                <div className="flex mt-[20px] lg:w-[50%] mx-auto">
                                 
                                      <button
                                            type="submit"
                                            form="myForm"
                                            className="p-2 text-center mx-auto rounded-full bg-gradient-to-r from-[#564740] to-[#887161] text-white  text-[12pt] uppercase  cursor-pointer hover:text-white hover:from-blue-700 hover:to-blue-500 transition duration-500 delay-1000 ease-in-out w-[120px]"
                                            alt="submit Checkout"
                                            onClick={handleSubmit}
                                          >
                                            save
                                          </button>
                                  
                                        <button
                                                type="submit"
                                                form="myForm"
                                                className="p-2 text-center mx-auto rounded-full bg-gradient-to-r from-[#564740] to-[#887161] text-white text-[12pt] uppercase  cursor-pointer hover:text-white hover:from-blue-700 hover:to-blue-500 transition duration-500 delay-1000 ease-in-out w-[120px]"
                                                alt="submit Checkout"
                                                onClick={handleSubmit}
                                              >
                                                print
                                              </button>
                                 
                                        <button
                                                type="submit"
                                                form="myForm"
                                                className="p-2 text-center mx-auto rounded-full bg-gradient-to-r from-[#564740] to-[#887161] text-white  text-[12pt] uppercase  cursor-pointer hover:text-white hover:from-blue-700 hover:to-blue-500 transition duration-500 delay-1000 ease-in-out w-[120px]"
                                                alt="submit Checkout"
                                                onClick={handleSubmit}
                                              >
                                                share
                                              </button>
                                  
                </div>

              </div>
            </div>}


        </div>
              
      </div>
    </div>
  );
};
