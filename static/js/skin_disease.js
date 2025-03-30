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
            // document.querySelector('.results-section').style.display = 'none';  // Hide results when uploading new image
        }
        reader.readAsDataURL(file);
    }
});


document.getElementById('analyze-btn').addEventListener('click', analyzeSkinCondition);

function analyzeSkinCondition() {
    const imageInput = document.getElementById('image-upload');
    if (!imageInput.files[0]) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    fetch('/predict_skin', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
    });
}

function displayResults(data) {
    // document.querySelector('.results-section').style.display = 'block';
    document.getElementById('disease').textContent = data.disease;
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';

    const topPredictionsList = document.getElementById('top-predictions');
    topPredictionsList.innerHTML = '';

    const alternativePredictions = data.topPredictions.filter(prediction => prediction.disease !== data.disease);
    
    alternativePredictions.forEach(prediction => {
        const li = document.createElement('li');
        li.textContent = `${prediction.disease}: ${(prediction.confidence * 100).toFixed(2)}%`;
        topPredictionsList.appendChild(li);
    });
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
    uploadArea.classList.add('highlight');
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.classList.remove('highlight');
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
            // document.querySelector('.results-section').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}
