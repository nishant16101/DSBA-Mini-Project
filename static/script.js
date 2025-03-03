document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    console.log('Form submitted');
    
    var fileInput = document.getElementById('image-upload');
    var file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file first');
        return;
    }
    
    console.log('File selected:', file.name);
    
    // Convert image to base64
    var reader = new FileReader();
    reader.onloadend = function() {
        var base64Image = reader.result;
        console.log('Image converted to base64');
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Received data:', data);
            displayResults(data, file);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = '<p class="error">Error processing image. Please try again.</p>';
        });
    };
    reader.readAsDataURL(file);
});

function displayResults(data, file) {
    console.log('Displaying results');
    
    // Display the uploaded image
    var uploadedImage = document.getElementById('uploaded-image');
    uploadedImage.style.display = 'block';
    uploadedImage.src = URL.createObjectURL(file);
    
    // Find region with highest probability
    let highestRegion = Object.keys(data).reduce((a, b) => data[a] > data[b] ? a : b);
    console.log('Highest region:', highestRegion);
    
    // Chinese face mapping database
    const faceMapping = {
        'forehead': {
            'organs': 'Bladder and Small Intestine',
            'issues': 'Poor digestion, stress, irregular sleep, toxin buildup',
            'suggestions': 'Improve hydration, reduce processed foods, manage stress, ensure adequate sleep'
        },
        'cheeks': {
            'organs': 'Stomach, Spleen, and Respiratory System',
            'issues': 'Poor diet, allergies, smoking, pollution exposure',
            'suggestions': 'Clean your phone regularly, avoid touching your face, improve diet with antioxidants'
        },
        'nose': {
            'organs': 'Heart and Blood Pressure',
            'issues': 'Blood pressure issues, poor circulation, vitamin B deficiency',
            'suggestions': 'Check blood pressure, consume vitamin B-rich foods, reduce sodium intake'
        },
        'chin': {
            'organs': 'Hormonal Balance and Reproductive System',
            'issues': 'Hormonal imbalance, stress, poor diet',
            'suggestions': 'Balance hormones with regular exercise, manage stress, consider gynecological check-up'
        }
    };
    
    // Create results HTML
    var resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<h3>Acne Region Analysis:</h3>';
    
    // Show all region probabilities
    for (var region in data) {
        let percentage = (data[region] * 100).toFixed(2);
        let isHighest = region === highestRegion;
        resultDiv.innerHTML += `<p class="${isHighest ? 'highest-region' : ''}">${region}: ${percentage}%${isHighest ? ' (Highest)' : ''}</p>`;
    }
    
    // Display Chinese face mapping for highest region
    resultDiv.innerHTML += `
        <div class="face-mapping">
            <h3>Face Mapping Analysis:</h3>
            <p><strong>Based on your ${highestRegion} acne, this may indicate:</strong></p>
            <p><strong>Related Body Systems:</strong> ${faceMapping[highestRegion].organs}</p>
            <p><strong>Potential Issues:</strong> ${faceMapping[highestRegion].issues}</p>
            <p><strong>Suggestions:</strong> ${faceMapping[highestRegion].suggestions}</p>
        </div>
    `;
}
