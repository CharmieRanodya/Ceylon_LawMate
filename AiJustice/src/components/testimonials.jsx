import React, { useState } from "react";
import myGif from "../images/gif.gif";
import axios from "axios";
import BarLoader from "react-spinners/ClipLoader";
import { Navigation } from "../components/navigation";

export const Testimonials = (props) => {
  const [commentText, setCommentText] = useState("");
  const [isLoading, setIsLoading] = React.useState(true);
 
  const [response, setResponse] = useState({
    "answer": "-",
    "reference": "-"
});

  const handleOnSubmit = (event) => {
    event.preventDefault();
    setIsLoading(false)
    console.log(commentText);
    axios
      .post(
        "http://ec2-13-229-183-94.ap-southeast-1.compute.amazonaws.com:5006/qna",
        {
          question:
          commentText,
        }
      )
      .then((response) => {
        console.log(
          "🚀 ~ file: testimonials.jsx:15 ~ .then ~ response:",
          response
        );
        setIsLoading(true)
        setResponse(response.data);
      });
  };

  return (
  
    <div className="main-body h-screen ">
      <div className="body-container h-screen  text-center relative object-fill">
          <div className="login__container absolute bg-opacity-30 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col w-[90%] space-y-4 rounded-lg">
            


          <textarea
              name="commentTextArea"
              type="text"
              className="resize-none p-5 h-[300px] border border-black"
              id="CommentsOrAdditionalInformation"
              placeholder="Type here.."
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
            />



            <button type="submit" form="myForm" className=" w-2/12 rounded-full text-[14pt] text-white uppercase p-5 font-bold bg-gradient-to-r from-[#564740] to-[#887161]" alt="submit Checkout" onClick={handleOnSubmit}>
              submit
            </button>


      <BarLoader loading={!isLoading} height={1} width={1} color="#36d7b7" />

            {response&&<div style={{ backgroundColor: "#F6ECE8" }}>
              <div className="testimonial text-left">
                <div className="testimonial-content">
                  <div className="testimonial-meta">Answer: {response?.answer}</div>
                  <p style={{marginTop:'10px'}}> Reference: {response?.reference}</p>
                </div>
              </div>
            </div>}
          </div>

      </div>
      </div>
    
  );
};
