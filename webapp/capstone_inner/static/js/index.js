var loadFile = function (event) {
    
    // Changing display object
    if (!document.getElementById('canvas').classList.contains('d-none')) {
        document.getElementById('canvas').classList.add('d-none');
    }
    if (!document.getElementById('videoPlaceHolder').classList.contains('d-none')) {
        document.getElementById('videoPlaceHolder').classList.add('d-none');
    }
    if (document.getElementById('placeholder').classList.contains('d-none')) {
        document.getElementById('placeholder').classList.remove('d-none');
    }

    // Displaying the selected image
    let placeholder = document.getElementById('placeholder');
    placeholder.src = URL.createObjectURL(event.target.files[0]);
    placeholder.onload = function () {
        URL.revokeObjectURL(placeholder.src) // free memory
    }
};

$("#camera").on("click", function () {

    // Changing display object
    if (!document.getElementById('canvas').classList.contains('d-none')) {
        document.getElementById('canvas').classList.add('d-none');
    }
    if (document.getElementById('videoPlaceHolder').classList.contains('d-none')) {
        document.getElementById('videoPlaceHolder').classList.remove('d-none');
    }
    if (!document.getElementById('placeholder').classList.contains('d-none')) {
        document.getElementById('placeholder').classList.add('d-none');
    }

    // Changing buttons
    document.getElementById('camera').classList.add('d-none');
    document.getElementById('capture').classList.remove('d-none');

    // Starting webcame and rendering it on screen
    const constraints = {
        video: true
    };
    const video = document.querySelector('video');
    navigator.mediaDevices.getUserMedia(constraints).
        then((stream) => { video.srcObject = stream });
});

$("#capture").on("click", function () {

    // Changing display object
    document.getElementById('canvas').classList.remove('d-none');
    document.getElementById('videoPlaceHolder').classList.add('d-none');

    // Changing buttons
    document.getElementById('camera').classList.remove('d-none');
    document.getElementById('capture').classList.add('d-none');

    // Drawing snapshot on canvas
    const context = canvas.getContext('2d');
    const video = document.querySelector('video');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Putting image into tag
    let canvass = document.getElementById('canvas');
    const imgForm = document.getElementById('img-form');
    var newInput = document.createElement('input');
    newInput.name = 'file2';
    newInput.type = 'text';
    newInput.className = 'd-none';
    newInput.value = canvass.toDataURL();
    imgForm.appendChild(newInput);

    // Stopping webcam
    video.srcObject.getVideoTracks().forEach(track => track.stop());
});