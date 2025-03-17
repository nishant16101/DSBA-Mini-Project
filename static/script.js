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
    
    // Diet suggestions based on acne region
    const dietSuggestions = {
        'forehead': {
            'title': 'Small Intestine (Pitta Imbalance)',
            'description': 'An overheated or inflamed intestine can cause acne.',
            'foods': [
                { type: 'Cooling Foods', items: 'Cucumber, coriander, fennel, coconut water' },
                { type: 'Herbs', items: 'Triphala, licorice, aloe vera juice' },
                { type: 'Easily Digestible Foods', items: 'Moong dal, rice, ghee, warm soups' }
            ]
        },
        'cheeks': {
            'title': 'Stomach & Respiratory System (Vata + Pitta Imbalance)',
            'description': 'Poor digestion and allergies can lead to acne.',
            'foods': [
                { type: 'Digestive Boosters', items: 'Buttermilk, ginger tea, cumin, ajwain' },
                { type: 'Fiber-Rich Foods', items: 'Psyllium husk (Isabgol), flaxseeds, triphala churna' },
                { type: 'Soothing Foods', items: 'Ghee, rice, moong dal, pumpkin' }
            ]
        },
        'nose': {
            'title': 'Heart (Pitta Imbalance)',
            'description': 'Poor circulation can lead to toxin buildup, causing acne.',
            'foods': [
                { type: 'Heart-Friendly Foods', items: 'Pomegranate, beetroot, almonds, walnuts' },
                { type: 'Cooling Herbs', items: 'Arjuna, Brahmi, Tulsi tea' },
                { type: 'Healthy Fats', items: 'Cow ghee, sesame seeds, flaxseeds' }
            ]
        },
        'chin': {
            'title': 'Reproductive System (Vata + Pitta Imbalance)',
            'description': 'Hormonal imbalances can cause cystic acne.',
            'foods': [
                { type: 'Hormone Balancing Foods', items: 'Leafy greens, omega-3 rich foods' },
                { type: 'Stress Reducing Herbs', items: 'Ashwagandha, Brahmi' },
                { type: 'Hydrating Foods', items: 'Watermelon, cucumber, coconut water' }
            ]
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
    
    // Display diet suggestions for highest region
    resultDiv.innerHTML += `
        <div class="diet-suggestions">
            <h3>Diet Suggestions for ${highestRegion} Acne:</h3>
            <h4>${dietSuggestions[highestRegion].title}</h4>
            <p>${dietSuggestions[highestRegion].description}</p>
            <ul>
                ${dietSuggestions[highestRegion].foods.map(food => `<li><strong>${food.type}:</strong> ${food.items}</li>`).join('')}
            </ul>
        </div>
    `;
}
