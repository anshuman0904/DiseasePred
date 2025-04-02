document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const modal = document.getElementById('resultModal');
    const closeButton = document.querySelector('.close-button');
    const newAssessmentBtn = document.getElementById('newAssessmentBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const resultDate = document.getElementById('resultDate');
    const riskIndicator = document.getElementById('riskIndicator');
    const riskValue = document.getElementById('riskValue');
    const resultDescription = document.getElementById('resultDescription');
    const recommendationList = document.getElementById('recommendationList');
    
    // Format current date
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(this);
        const processedData = {};
        
        // Process form data to match model's expected format
        for (let [key, value] of formData.entries()) {
            if (key === 'gender') {
                // Flip gender values: form has male=1, female=0 but model expects male=0, female=1
                processedData[key] = value === '1' ? 0 : 1;
            } else if (key === 'age') {
                // Keep age as is, but ensure it's a number
                processedData[key] = parseInt(value, 10);
            } else {
                // Convert Yes/No values: form has yes=1, no=0 but model expects yes=2, no=1
                processedData[key] = value === '1' ? 2 : 1;
            }
        }
        
        fetch('/predict_lungs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(processedData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            displayError(error.message);
        });
    });
    
    function displayResults(data) {
        // Set the date
        resultDate.textContent = formattedDate;
        
        // Set risk indicator based on prediction
        if (data.prediction === 'YES') {
            setHighRisk();
        } else {
            setLowRisk();
        }
        
        // Show the modal
        modal.style.display = 'block';
    }
    
    function setHighRisk() {
        riskIndicator.className = 'result-indicator high-risk';
        riskValue.textContent = 'Elevated Risk';
        resultDescription.innerHTML = '<p>Based on the information provided, our analysis indicates an <strong>elevated risk</strong> for lung cancer. This is not a diagnosis but a recommendation to consult with a healthcare professional promptly.</p>';
        
        recommendationList.innerHTML = `
            <li>Schedule an immediate consultation with a pulmonologist</li>
            <li>Consider chest imaging tests as recommended by your doctor</li>
            <li>If you smoke, discuss smoking cessation programs with your healthcare provider</li>
            <li>Reduce exposure to secondhand smoke and other environmental toxins</li>
            <li>Monitor your symptoms closely and seek prompt medical attention for any changes</li>
        `;
    }
    
    function setLowRisk() {
        riskIndicator.className = 'result-indicator low-risk';
        riskValue.textContent = 'Lower Risk';
        resultDescription.innerHTML = '<p>Based on the information provided, our analysis indicates a <strong>lower risk</strong> for lung cancer. However, maintaining a healthy lifestyle and being aware of respiratory symptoms is always recommended.</p>';
        
        recommendationList.innerHTML = `
            <li>Maintain a smoke-free lifestyle</li>
            <li>Minimize exposure to environmental pollutants and toxins</li>
            <li>Consider regular lung health check-ups, especially if you have risk factors</li>
            <li>Stay physically active and maintain a balanced diet rich in antioxidants</li>
            <li>Be aware of changes in respiratory health and report persistent symptoms to your doctor</li>
        `;
    }
    
    function displayError(message) {
        resultDescription.innerHTML = `<p>Error: ${message}</p>`;
        modal.style.display = 'block';
    }
    
    // Close the modal when clicking the close button
    closeButton.addEventListener('click', closeModal);
    
    // Close the modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            closeModal();
        }
    });
    
    function closeModal() {
        modal.style.display = 'none';
    }
    
    // New assessment button
    newAssessmentBtn.addEventListener('click', function() {
        form.reset();
        closeModal();
    });
    
    // Download button (in a real application, this would generate a PDF)
    downloadBtn.addEventListener('click', function() {
        alert('In a real application, this would download a PDF report of your lung cancer risk assessment.');
    });
});
