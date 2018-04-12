function generateThumbnail() {
  uploadImage('lol','', 'lol');
}

var convertToBase64 = function(url, imagetype, callback){

  var img = document.createElement('IMG');
  var c = document.createElement("canvas");
  var ctx = c.getContext("2d");
  var dataURL= ''; 
  img.crossOrigin = 'Anonymous';
  c.width = 500;
  c.height = 375;
  ctx.drawImage(video, 0, 0, 500, 375);
  thumbs.appendChild(c);
  dataURL = c.toDataURL();
  callback(dataURL);
  img.src = url;

};

var sendBase64ToServer = function(name, base64){
    var sendData = JSON.stringify({image: base64});
    //alert(sendData);

    var path = "http://127.0.0.1:8080/uploadImage/";
  
  // document.getElementById('testing').innerHTML = sendData;
    $.ajax({
            url: path,
            data: sendData,
            contentType : 'application/json',
            type: 'POST',
            success: function(response) {
                alert(response.emotion);
            },
            error: function(error) {
                alert(error);
            }
        });

};


var uploadImage = function(src, name, type){
    convertToBase64(src, type, function(data){
        sendBase64ToServer(name, data);
    });
};

var video = document.querySelector("#videoElement");
 
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;
 
if (navigator.getUserMedia) {       
    navigator.getUserMedia({video: true}, handleVideo, videoError);
}
 
function handleVideo(stream) {
    video.src = window.URL.createObjectURL(stream);
}
 
function videoError(e) {
    // do something
}
