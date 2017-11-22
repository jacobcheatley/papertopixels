'use strict';

window.onload = function() {
    // Variables
    var videoDevices = [];
    var currentDevice = 0;
    var videoElement = document.querySelector('video');

    // Start of running
    navigator.mediaDevices
        .enumerateDevices()
        .then(gotDevices)
        .then(getStream)
        .catch(handleError);

    // Functions
    function gotDevices(deviceInfos) {
        for (var i = 0; i < deviceInfos.length; i++) {
            var deviceInfo = deviceInfos[i];
            if (deviceInfo.kind === 'videoinput') {
                videoDevices.push(deviceInfo.deviceId);
            }
        }
    }

    function getStream() {
        if (window.stream) {
            window.stream.getTracks().forEach(function (track) {
                track.stop();
            });
        }

        var constraints = {
            video: {deviceId: {exact: videoDevices[currentDevice]}}
        };

        navigator.mediaDevices.getUserMedia(constraints).then(gotStream).catch(handleError);
    }

    function gotStream(stream) {
        window.stream = stream;
        videoElement.srcObject = stream;
        videoElement.play();
    }

    function switchSource() {
        console.log('SWITCH');
        currentDevice = (currentDevice + 1) % videoDevices.length;
        getStream();
    }

    function handleError(error) {
        console.log('Error: ', error);
    }

    // Event handlers
    videoElement.addEventListener('click', switchSource, false);
};