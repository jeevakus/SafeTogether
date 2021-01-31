const webcamElement= document.getElementById('webcam');
let net;
let isPredicting = false;
function startPredicting(){
 isPredicting=true;
 app();
}
function stopPredicting(){
 isPredicting=false;
 app();
}
async function app(){
 console.log('Loading model..');
 net= await tf.automl.loadImageClassification('model-export_icn_tf_js-Hands_Test_20210130073643-2021-01-31T03_00_19.340343Z_model.json.json');
 console.log('Successfully loaded model');
 
 const webcam = await tf.data.webcam(webcamElement);
 while(isPredicting){
 const img = await webcam.capture();
 const result = await net.classify(img);
 
 console.log(result);
 
 document.getElementById("Thumb_In_Fist").innerText=result['0']['label']+": "+Math.round(result['0']['prob']*100)+"%";
 document.getElementById("Four_Fingers_Up").innerText=result['1']['label']+": "+Math.round(result['1']['prob']*100)+"%";
img.dispose();
 
await tf.nextFrame();
 
 }
 
}