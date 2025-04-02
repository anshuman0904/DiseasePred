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
        
        fetch('/predict_diabetes', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Set the date
            resultDate.textContent = formattedDate;
            
            // Set risk indicator based on prediction
            if (data.prediction === 'Positive')
				{
                riskIndicator.className = 'result-indicator high-risk';
                riskValue.textContent = 'Elevated Risk';
                resultDescription.innerHTML = '<p>Based on the information provided, our analysis indicates an <strong>elevated risk</strong> for diabetes. This is not a diagnosis but a recommendation to consult with a healthcare professional.</p>';
                
                recommendationList.innerHTML = `
                    <li>Schedule a consultation with an endocrinologist</li>
                    <li>Consider blood glucose testing</li>
                    <li>Review your dietary habits with a nutritionist</li>
                    <li>Implement a regular exercise routine</li>
                `;
            } else {
                riskIndicator.className = 'result-indicator low-risk';
                riskValue.textContent = 'Lower Risk';
                resultDescription.innerHTML = '<p>Based on the information provided, our analysis indicates a <strong>lower risk</strong> for diabetes. However, maintaining a healthy lifestyle is always recommended.</p>';
                
                recommendationList.innerHTML = `
                    <li>Maintain a balanced diet rich in whole foods</li>
                    <li>Continue regular physical activity</li>
                    <li>Consider annual health check-ups</li>
                    <li>Stay hydrated and manage stress levels</li>
                `;
            }
            
            // Show the modal
            modal.style.display = 'block';
        })
        .catch(error => {
            resultDescription.innerHTML = `<p>Error: ${error.message}</p>`;
            modal.style.display = 'block';
        });
    });
    
    // Close the modal when clicking the close button
    closeButton.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    
    // Close the modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // New assessment button
    newAssessmentBtn.addEventListener('click', function() {
        form.reset();
        modal.style.display = 'none';
    });
    
    // Download button (in a real application, this would generate a PDF)
    downloadBtn.addEventListener('click', function() {
        alert('In a real application, this would download a PDF report of your results.');
    });
});
