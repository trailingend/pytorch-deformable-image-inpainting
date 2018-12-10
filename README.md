# Image Inpainting Demo
### Authors: Lynn Duan, Lei Pan, Jenna Zhang, Jayleen Zhou

-----------------------------

## sitemap inside the working folder
├── README.md
├── app.py
├── server/
    ├── torch_deform_conv/
    ├── predict.py
    ├── <model_file>
    ├── models.py
    ├── layers.py
    └── utils.html
└── static/
    ├── dist/
    ├── script/
        ├── index.jsx
        └── main.jsx
    ├── style/
        └── main.css
    └── index.html

-----------------------------

## How to run this demo locally
- in the project root folder, run `python app.py` in Terminal
- url: localhost:5000

## To compile front-end
- change directory into the static folder, run `npm run dev` in Terminal
- url: localhost:8080
- remember to clear browser cache

## How to test front-end framework
- change directory into the static folder, run `npm install` and then `npm run watch` in Terminal

-----------------------------

<!-- ## Heroku Information
- how to create:
    heroku create fgsocialtest
- how to update:
    git push heroku master
- App name: fg-social-api
- urls: https://fg-social-api.herokuapp.com/ | https://git.heroku.com/fg-social-api.git -->

-----------------------------

## Approaches
- Front-end: using ReactJS to render the front-end buttons, jQuery to handle AJAX communications
- Back-end: using Python in Flask framework to serve as the back-end server.
<!-- - Online: locally the app is instantiated via terminal by calling the Flask app. On Heroku, the app is wrapped with the web service gunicorn to initiate the Flask server. -->

-----------------------------
## In main.jsx
// <div className="img-mask" onClick={this.cropARegion}>
//     <div className="dot-mask"></div>
//     <div className="rect-mask"></div>
// </div>
