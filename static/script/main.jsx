import "../style/main.css";
import React from "react";
import inputSrc from "../../server/input.jpg";

var $ = require('jquery');

/***
 *  Class to capture user interaction and communicate with backend for image inpainting
***/
export default class Main extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            message: 'Upload a photo to start',
            type: "dummy",
            imgSrc: inputSrc,
            rectStatus: 0,
            mouse_pos: [0, 0, 0, 0],
            snapStatus: 0,
            snapText: "Take a Photo"
        };
        this.setDisplayMessage = this.setDisplayMessage.bind(this);
        this.uploadPhoto = this.uploadPhoto.bind(this);
        this.capturePhoto = this.capturePhoto.bind(this);
        this.generatePhoto = this.generatePhoto.bind(this);
        this.cropARegion = this.cropARegion.bind(this);
        this.initMask = this.initMask.bind(this);
        this.endMask = this.endMask.bind(this);
        this.cropMask = this.cropMask.bind(this);
    }

    /*
        Method to change display text
    */
    setDisplayMessage(msg) {
        this.setState({message: msg});
    }

    /*
        Method to change image source
    */
    setImageSource(url) {
        this.setState({imgSrc: url});
    }

    /*
        Method to upload photo from local to web
    */
    uploadPhoto(event) {
        event.preventDefault();

        this.setDisplayMessage("Uploading...");

        var reader  = new FileReader();
        var file = event.target.files[0];

        reader.addEventListener("load", () => {
            var dataURL = reader.result
            this.setState({
                type: "jpg",
                imgSrc: dataURL
            });
            this.setDisplayMessage("Select a region on the sample’s face to regenerate");
            console.log($('.img-elem').width() + " " + document.getElementsByClassName('img-elem')[0].height)
            $('.img-elem').css('marginTop', - ($('.img-elem').height() - $('.img-elem').width()) / 2);
        }, false);

        if (file) {
            reader.readAsDataURL(file);
        }
    }

    /*
        Method to capture photo from camera
    */
    capturePhoto() {
        if (this.state.snapStatus == 0) {
            this.setState({
                snapText: 'Capture',
                snapStatus: 1,
                message: "Capturing photo from camera"
            });
            var video = document.getElementById('img-video');
            $('.img-video').css('opacity', 1);
            if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
                    this.state.videoStream = stream;
                    video.srcObject = stream;
                    video.play();
                });
            }
        } else if (this.state.snapStatus == 1){
            var video = document.getElementById('img-video');
            var canvas = document.getElementById('img-stream');
            var context = canvas.getContext('2d');
            var videoHeight = this.state.videoStream.getTracks()[0].getSettings().height;
            var videoWidth = this.state.videoStream.getTracks()[0].getSettings().width;
            canvas.width = 300;
            canvas.height = 300;
            context.drawImage(video, (videoWidth - videoHeight) / 2, 0, videoHeight, videoHeight, 0, 0, 300, 300);
            setTimeout(()=> {
                video.pause();
                video.src = "";
                this.state.videoStream.getTracks()[0].stop();
            }, 500);

            $('.img-video').css('opacity', 0);
            this.setState({
                type: "png",
                imgSrc: canvas.toDataURL('image/png'),
                snapText: 'Take a Photo',
                snapStatus: 0,
                message: "Photo captured"
            });
        }
    }

    /*
        Method to communicate with server and receive generated photo from server
    */
    generatePhoto() {
        var base64result;
        if (this.state.type == "jpg" || this.state.type == "png") {
            if (this.state.rectStatus == 2) {
                var mask = document.getElementById('img-mask');
                base64result = this.state.imgSrc.split(',')[1];
                this.setDisplayMessage("Communicating with server and generating new face");
                $.getJSON($SCRIPT_ROOT + '/_generate', {
                    startUploading: true,
                    input: base64result, // this.state.imgSrc,
                    type: this.state.type,
                    mask: mask.toDataURL('image/png').split(',')[1],
                    // mask1: this.state.mouse_pos[0],
                    // mask2:this.state.mouse_pos[1],
                    // mask3: this.state.mouse_pos[2],
                    // mask4:this.state.mouse_pos[3],
                }, (data) => {
                    $('.img-mask').css( { 'opacity': 0 });
                    // $('.img-elem').css( { 'width': 'auto' });
                    $('.img-group').css( { 'width': '600px' });
                    this.setDisplayMessage(data.msg);
                    this.setImageSource(data.result);
                    console.log("communication success")
                });
            } else {
                this.setDisplayMessage("Please draw a mask on the photo");
            }
        } else if (this.state.type == "dummy") {
            this.setDisplayMessage("Upload a photo to start");
        }

    }

    /*
        Method to create mask
    */
    cropARegion(e) {
        if (this.state.type != "dummy") {
            var left_padding = $('.img-group').css("margin-left");
            var left_padding = left_padding.substring(0, left_padding.length - 2);
            var total_left = (e.pageX + window.pageXOffset);
            var mouseX = total_left - left_padding;

            var bodyRect = document.body.getBoundingClientRect()
            var elemRect = document.getElementsByClassName("img-mask")[0].getBoundingClientRect();
            var elem_top   = elemRect.top - bodyRect.top;
            var total_top = e.pageY + window.pageYOffset;;
            var mouseY = total_top - elem_top;

            if (this.state.rectStatus == 0) {
                var mouse_first_pos = [mouseX, mouseY, 0, 0];
                this.setState({
                    rectStatus: 1,
                    mouse_pos: mouse_first_pos,
                    message: "Top left corner of mask selected"
                });
                $('.img-mask').css( 'cursor', 'crosshair' );
                $('.dot-mask').css( {'left': mouseX, 'top': mouseY, 'opacity': 1});
                $('.rect-mask').css( {'left': mouseX, 'top': mouseY, 'opacity': 0});
            } else if (this.state.rectStatus == 1){
                var mouse_second_pos = this.state.mouse_pos;
                mouse_second_pos[2] = mouseX;
                mouse_second_pos[3] = mouseY;
                this.setState({
                    rectStatus: 2,
                    mouse_pos: mouse_second_pos,
                    message: "Mask completed. Click generate to proceed"
                });
                $('.img-mask').css( 'cursor', 'pointer' );
                $('.dot-mask').css( {'opacity': 0});
                $('.rect-mask').css( {
                    'width': mouse_second_pos[2] - mouse_second_pos[0],
                    'height': mouse_second_pos[3] - mouse_second_pos[1],
                    'opacity': 1
                });
            } else {
                this.setDisplayMessage("Mask already generated");
            }
        } else {
            this.setDisplayMessage("You need to upload a photo to start");
        }

    }

    /*
        Method to start mask painting
    */
    initMask(e) {
        if (this.state.type != "dummy") {
            if (this.state.rectStatus == 0 || this.state.rectStatus == 2) {
                this.setState({rectStatus: 1});
            }
        }
    }

    /*
        Method to end mask painting
    */
    endMask(e) {
        if (this.state.rectStatus == 1) {
            this.setState({rectStatus: 2});
        }
    }

    /*
        Method to paint mask
    */
    cropMask(e) {
        var left_padding = $('.img-group').css("margin-left");
        var left_padding = left_padding.substring(0, left_padding.length - 2);
        var total_left = (e.pageX + window.pageXOffset);
        var mouseX = total_left - left_padding;

        var bodyRect = document.body.getBoundingClientRect()
        var elemRect = document.getElementsByClassName("img-mask")[0].getBoundingClientRect();
        var elem_top   = elemRect.top - bodyRect.top;
        var total_top = e.pageY + window.pageYOffset;;
        var mouseY = total_top - elem_top;
        if (this.state.type != "dummy") {
            if (this.state.rectStatus == 1){
                const ctx = this.refs.mask.getContext('2d');
                ctx.fillStyle = '#808080';
                ctx.beginPath();
                ctx.arc(mouseX, mouseY, 10, 0, 2 * Math.PI);
                ctx.fill();
            }
        } else {
            this.setDisplayMessage("You need to upload a photo to start");
        }
    }

    /*
        Method to render dom
    */
    render () {
        return <div className="container">
            <h1 className="ttl-group">Image Inpainting</h1>
            <div className="msg-group">
                <p>{this.state.message}</p>
            </div>
            <div className="img-group">
                <img src={this.state.imgSrc} className="img-elem"/>
                <video id="img-video" className="img-video" autoPlay></video>
                <canvas id="img-stream" className="img-stream" width="300" height="300"></canvas>
                <canvas id="img-mask" ref="mask" className="img-mask" width="300" height="300" onMouseDown={this.initMask} onMouseMove={this.cropMask} onMouseUp={this.endMask} ></canvas>
            </div>
            <div className="btn-groups">
                <label htmlFor="action-upload" className="action-label">Upload a Photo</label>
                <input type="file" id="action-upload" className="action-btn" onChange={this.uploadPhoto}/>
                <button id="action-capture" className="action-btn" onClick={this.capturePhoto}>{this.state.snapText}</button>
                <button id="action-generate" className="action-btn" onClick={this.generatePhoto}>Generate</button>
            </div>
        </div>;
    }
}
