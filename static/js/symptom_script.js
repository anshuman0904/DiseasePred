const symptomsInput = document.getElementById('symptoms');
const speechBtn = document.getElementById('speech-btn');
const predictBtn = document.getElementById('predict-btn');
const diseaseOutput = document.getElementById('disease');
const confidenceOutput = document.getElementById('confidence');
const topPredictionsOutput = document.getElementById('top-predictions');
const toggleVoiceBtn = document.getElementById('toggle-voice-btn');
const languageSelect = document.getElementById('language-select');

let voiceEnabled = true;

// Toggle Voice Output
toggleVoiceBtn.addEventListener('click', () => {
    voiceEnabled = !voiceEnabled;
    toggleVoiceBtn.innerHTML = voiceEnabled ? 
        "<span class='icon'>🔊</span> Voice Output" : 
        "<span class='icon'>🔇</span> Voice Output";
});

// Speech Recognition with Language Selection
speechBtn.addEventListener('click', () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = languageSelect.value;
    recognition.onresult = event => {
        symptomsInput.value = event.results[0][0].transcript;
    };
    recognition.start();
});

// Function to Speak the Predicted Disease
function speakText(text, lang = "en-US") {
    if (voiceEnabled) {
        const speech = new SpeechSynthesisUtterance(text);
        speech.lang = lang;
        speech.volume = 1;
        window.speechSynthesis.speak(speech);
    }
}

// Predict Disease
predictBtn.addEventListener('click', async () => {
    const symptoms = symptomsInput.value.trim();
    if (!symptoms) {
        alert('Please enter symptoms before analyzing.');
        return;
    }

    // Show loading state
    diseaseOutput.textContent = "Analyzing...";
    confidenceOutput.textContent = "-";
    topPredictionsOutput.innerHTML = '<li class="placeholder">Processing your symptoms...</li>';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms })
        });

        const result = await response.json();
        const diseaseText = result.disease || "Not Available";
        const confidenceText = `${(result.confidence * 100).toFixed(2)}%`;

        diseaseOutput.textContent = diseaseText;
        confidenceOutput.textContent = confidenceText;

        // Clear previous predictions
        topPredictionsOutput.innerHTML = '';
        
        // Filter out the primary diagnosis from alternative possibilities
        const alternativePredictions = result.topPredictions.filter(
            prediction => prediction.disease !== diseaseText
        );
        
        if (alternativePredictions.length > 0) {
            alternativePredictions.forEach(prediction => {
                const li = document.createElement('li');
                li.textContent = `${prediction.disease} - Confidence: ${(prediction.confidence * 100).toFixed(2)}%`;
                topPredictionsOutput.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.className = 'placeholder';
            li.textContent = 'No alternative diagnoses available';
            topPredictionsOutput.appendChild(li);
        }

        // Speak the result in selected language
        speakText(`Primary diagnosis is ${diseaseText}. Confidence level is ${confidenceText}`, languageSelect.value);

    } catch (error) {
        console.error('Error predicting disease:', error);
        diseaseOutput.textContent = "Analysis Failed";
        confidenceOutput.textContent = "-";
        topPredictionsOutput.innerHTML = '<li class="placeholder">Unable to process your request. Please try again later.</li>';
        alert('Failed to analyze symptoms. Please try again later.');
    }
});
