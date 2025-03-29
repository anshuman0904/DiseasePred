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
    toggleVoiceBtn.textContent = voiceEnabled ? "🔊 Voice ON" : "🔇 Voice OFF";
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
        alert('Please enter symptoms before predicting.');
        return;
    }

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

        topPredictionsOutput.innerHTML = '';
        result.topPredictions.forEach(prediction => {
            const li = document.createElement('li');
            li.textContent = `${prediction.disease} - Confidence: ${(prediction.confidence * 100).toFixed(2)}%`;
            topPredictionsOutput.appendChild(li);
        });

        // Speak the result in selected language
        speakText(`Predicted disease is ${diseaseText}. Confidence level is ${confidenceText}`, languageSelect.value);

    } catch (error) {
        console.error('Error predicting disease:', error);
        alert('Failed to predict disease. Try again later.');
    }
});
