'use strict';

function addvector(a,b){
  return a.map((e,i) => e + b[i]);
}
module.exports.centroid = async (event) => {
  console.log(event)
  var body = JSON.parse(event['body'])
  // expect an array of strings representing vectors
  if (Array.isArray(body) && body.length >0) {
    const count = body.length;
    var centroid = JSON.parse(body[0]);

    for (var i = 1; i < count; i++) {
      centroid = addvector(centroid,JSON.parse(body[i]))
    }
 
    let norm = Math.sqrt(centroid.reduce((a,e) => a + e*e,0))
    centroid = centroid.map((e) => e/norm);
    var response = JSON.stringify(centroid)
  } else {
    var response = JSON.stringify(
      {
        message: 'Empty list',
        input: event,
      },
      null,
      2
    )
    
  }
  return {
    statusCode: 200,
    body: response,
  };

};
