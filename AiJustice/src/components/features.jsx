import React from "react";

export const Features = (props) => {
  return (
    <div  className="text-center mx-auto justify-between ">
      <div className="container mx-auto">
        <div className="col-md-12 mx-auto">
          <h2 className="small-text">Features</h2>
        </div>
        <div className="row mx-auto justify-between text-center " >
          {props.data
            ? props.data.map((d, i) => (
                <div key={`${d.title}-${i}`} className="col-md-3 col-xs-3 mx-auto justify-center text-center mt-5 text-[#887161]" >
                  {" "}
                  <i className={`${d.icon} text-[32pt] mb-5`}  ></i>
                  <p className="text-gray-400 ">{d.title}</p>
                  {/* <p>{d.text}</p> */}
                </div>
              ))
            : "Loading..."}
        </div>
      </div>
    </div>
  );
};
