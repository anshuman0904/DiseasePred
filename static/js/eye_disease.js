document.getElementById('upload-area').addEventListener('click', function() {
    document.getElementById('image-upload').click();
});

document.getElementById('image-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview-img').src = e.target.result;
            document.querySelector('.preview-section').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

// Add voice toggle button functionality
const toggleVoiceBtn = document.getElementById('toggle-voice-btn') || document.createElement('button');
let voiceEnabled = true;

if (toggleVoiceBtn.id === 'toggle-voice-btn') {
    toggleVoiceBtn.addEventListener("click", () => {
        voiceEnabled = !voiceEnabled;
        toggleVoiceBtn.innerHTML = voiceEnabled
            ? "<span class='icon'>🔊</span> Voice Output"
            : "<span class='icon'>🔇</span> Voice Output";
    });
}

// Function to speak the predicted disease
function speakText(text, lang = "en-US") {
    if (voiceEnabled) {
        const speech = new SpeechSynthesisUtterance(text);
        speech.lang = lang;
        speech.volume = 1;
        window.speechSynthesis.speak(speech);
    }
}

document.getElementById('analyze-btn').addEventListener('click', analyzeEyeCondition);

function analyzeEyeCondition() {
    const imageInput = document.getElementById('image-upload');
    if (!imageInput.files[0]) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    // Show loading state
    document.getElementById('disease').textContent = "Analyzing...";
    document.getElementById('confidence').textContent = "-";
    document.getElementById('top-predictions').innerHTML = 
        '<li class="placeholder">Processing your image...</li>';

    fetch('/predict_eye', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        displayResults(data);
        
        // Speak the result
        const diseaseText = data.disease || "Not Available";
        const confidenceText = `${(data.confidence * 100).toFixed(2)}%`;
        speakText(`Eye analysis complete. Primary diagnosis is ${diseaseText}. Confidence level is ${confidenceText}`);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('disease').textContent = "Analysis Failed";
        document.getElementById('confidence').textContent = "-";
        document.getElementById('top-predictions').innerHTML = 
            '<li class="placeholder">Unable to process your request. Please try again later.</li>';
        alert('An error occurred during analysis. Please try again.');
    });
}

function displayResults(data) {
    document.getElementById('disease').textContent = data.disease;
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';

    const topPredictionsList = document.getElementById('top-predictions');
    topPredictionsList.innerHTML = '';

    const alternativePredictions = data.topPredictions.filter(prediction => prediction.disease !== data.disease);
    
    if (alternativePredictions.length > 0) {
        alternativePredictions.forEach(prediction => {
            const li = document.createElement('li');
            li.textContent = `${prediction.disease}: ${(prediction.confidence * 100).toFixed(2)}%`;
            topPredictionsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = "placeholder";
        li.textContent = "No alternative diagnoses available";
        topPredictionsList.appendChild(li);
    }
}

// Drag and drop functionality
const uploadArea = document.getElementById('upload-area');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.add('highlight');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.remove('highlight');
    });
});

uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('image-upload').files = files;
        handleFiles(files);
    }
}

function handleFiles(files) {
    const file = files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview-img').src = e.target.result;
            document.querySelector('.preview-section').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}
