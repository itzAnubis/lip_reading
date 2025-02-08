let loggedInUsername = 'Guest'; // Default username

// Function to show a specific section and hide others
function showSection(section) {
    const sections = document.querySelectorAll('.container');
    sections.forEach((sec) => {
        sec.style.display = 'none'; // Hide all sections
    });
    document.getElementById(section).style.display = 'flex'; // Show the selected section
    updateProfile(); // Update profile display if necessary
}

// Functions for login and upload
function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    console.log(username);

    // Basic validation (you can replace this with actual authentication logic)
    if (username && password) {
        loggedInUsername = username; // Store the logged-in username
        alert(`Logged in as ${loggedInUsername}`);
        showSection('welcome'); // Redirect to welcome screen
    } else {
        alert("Please enter both username and password.");
    }
}

function uploadVideo() {
    const videoInput = document.getElementById('videoUpload');

    if (videoInput.files.length > 0) {
        const file = videoInput.files[0];

        // Create FormData to send the file
        const formData = new FormData();
        formData.append('file', file);

        // Send the video file to the Flask backend
        fetch('http://127.0.0.1:5050/predict_uploaded_video', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                const predictedClassLabel = data.predicted_class || "Unknown";
                console.log(data.predicted_class);
                alert(`Video processed successfully! Predicted words: ${predictedClassLabel}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while uploading the video.');
        });
    } else {
        alert("Please select a video file to upload.");
    }
}

// Update the profile section with the logged-in username
function updateProfile() {
    document.getElementById('profile-username').textContent = loggedInUsername;
}

// Live transcription functionality
let mediaRecorder;
let recordedChunks = [];

async function startCamera() {
    const videoElement = document.getElementById('videoElement');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
            recordedChunks.push(event.data);
            console.log('Recorded data available:', recordedChunks.length);
        };
        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');

            a.style.display = 'none';
            a.href = url;
            a.download = 'live_transcription.webm';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        };
    } catch (error) {
        console.error("Error accessing the camera: ", error);
    }
}

function startRecording() {
    if (mediaRecorder && mediaRecorder.state === "inactive") {
        recordedChunks = [];
        mediaRecorder.start();
        alert("Recording started!");
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        alert("Recording stopped!");
    }
}

// Initialize the welcome section
window.onload = () => {
    showSection('welcome'); // Show the welcome section on page load
    startCamera(); // Start the camera for live transcription
};
