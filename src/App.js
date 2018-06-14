import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";
import "./scss/app.scss";

class App extends Component {
  constructor(props) {
    super(props);
    this.webcamElement = React.createRef();
  }

  componentDidMount() {
    const navigatorAny = navigator;

    navigator.getUserMedia =
      navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia ||
      navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;

    if (navigator.getUserMedia) {
      navigator.getUserMedia(
        { video: true },
        stream => {
          this.webcamElement.srcObject = stream;
          this.webcamElement.addEventListener(
            "loadeddata",
            () => {
              console.log(
                this.webcamElement.videoWidth,
                this.webcamElement.videoHeight
              );
              this.adjustVideoSize(
                this.webcamElement.videoWidth,
                this.webcamElement.videoHeight
              );
            },
            false
          );
        },
        error => {
          console.log("Cannot start webcam");
        }
      );
    }
  }

  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  capture() {
    return tf.tidy(() => {
      const webcamImage = tf.fromPixels(this.webcamElement);
      const croppedImage = this.cropImage(webcamImage);
      const batchedImage = croppedImage.expandDims(0);

      batchedImage
        .toFloat()
        .div(tf.scalar(127))
        .sub(tf.scalar(1))
        .print();
    });
  }

  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  async loadMobilenet() {
    const mobilenet = await tf.loadModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer("conv_pw_13_relu");
    return tf.model({ inputs: this.model.inputs, outputs: layer.output });
  }

  render() {
    return (
      <div>
        <div className="container">
          <h1>Hello World!</h1>
          <div>
            <video
              ref={refs => {
                this.webcamElement = refs;
              }}
              autoPlay="true"
              width={300}
              height={400}
            />
          </div>
          <button onClick={() => this.capture()}>Capture</button>
        </div>
      </div>
    );
  }
}

export default App;
