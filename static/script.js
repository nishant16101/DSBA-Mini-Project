document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please upload an image first.');
        return;
    }

    const reader = new FileReader();
    
    reader.onload = function(event) {
        const imageData = event.target.result;
        const data = { image: imageData };
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML = `
                <h3>Acne Detection Results:</h3>
                <ul>
                    <li>Forehead: ${data.forehead}%</li>
                    <li>Cheeks: ${data.cheeks}%</li>
                    <li>Nose: ${data.nose}%</li>
                    <li>Chin: ${data.chin}%</li>
                </ul>
            `;
            document.getElementById('proceed-btn').style.display = 'block';
            
            // Store detected regions in local storage
            localStorage.setItem('detectedRegions', JSON.stringify(data));
        })
        .catch(error => console.error('Error:', error));
    };
    
    reader.readAsDataURL(file);
});

function showDiet() {
    window.location.href = "/suggestions";  // Use direct URL instead of template syntax
}

document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname.includes('suggestions')) {
        const dietSuggestions = document.getElementById('diet-suggestions');
        dietSuggestions.innerHTML = '';

        const detectedRegions = JSON.parse(localStorage.getItem('detectedRegions')) || {};

        if (!detectedRegions || Object.keys(detectedRegions).length === 0) {
            dietSuggestions.innerHTML = `<p>No acne detected or data unavailable.</p>`;
            return;
        }

        const regionSuggestions = {
            forehead: `
                <div class="diet-card">
                    <h3>Forehead Diet Suggestions</h3>
                    <p>Cooling Foods: Cucumber, coriander, fennel, coconut water.</p>
                    <p>Herbs: Triphala, licorice, aloe vera juice.</p>
                    <p>Easily Digestible Foods: Moong dal, rice, ghee, warm soups.</p>
                </div>
            `,
            cheeks: `
                <div class="diet-card">
                    <h3>Cheeks Diet Suggestions</h3>
                    <p>Diuretic Foods: Barley water, cranberry juice, coconut water.</p>
                    <p>Herbs: Punarnava, gokshura, coriander seeds tea.</p>
                    <p>Hydrating Foods: Watermelon, cucumber, ash gourd.</p>
                </div>
            `,
            nose: `
                <div class="diet-card">
                    <h3>Nose Diet Suggestions</h3>
                    <p>Heart-Friendly Foods: Pomegranate, beetroot, almonds, walnuts.</p>
                    <p>Cooling Herbs: Arjuna, Brahmi, Tulsi tea.</p>
                    <p>Healthy Fats: Cow ghee, sesame seeds, flaxseeds.</p>
                </div>
            `,
            chin: `
                <div class="diet-card">
                    <h3>Chin Diet Suggestions</h3>
                    <p>Liver Cleansing Foods: Turmeric, bitter gourd, neem, amla.</p>
                    <p>Herbs: Kutki, Bhumiamalaki, Aloe Vera.</p>
                    <p>Foods to Support Detox: Green leafy vegetables, beetroot, radish.</p>
                </div>
            `
        };

        Object.keys(regionSuggestions).forEach(region => {
            if (detectedRegions[region] > 50) {
                dietSuggestions.insertAdjacentHTML('beforeend', regionSuggestions[region]);
            }
        });

        // Additional holistic diet suggestions
        const extraSuggestions = [
            {
                condition: detectedRegions.forehead > 50,
                suggestion: `
                    <div class="diet-card">
                        <h3>Small Intestine Diet Suggestions</h3>
                        <p>Cooling Foods: Cucumber, coriander, fennel, coconut water.</p>
                        <p>Herbs: Triphala, licorice, aloe vera juice.</p>
                        <p>Easily Digestible Foods: Moong dal, rice, ghee, warm soups.</p>
                    </div>
                `
            },
            {
                condition: detectedRegions.cheeks > 50,
                suggestion: `
                    <div class="diet-card">
                        <h3>Bladder Diet Suggestions</h3>
                        <p>Diuretic Foods: Barley water, cranberry juice, coconut water.</p>
                        <p>Herbs: Punarnava, gokshura, coriander seeds tea.</p>
                        <p>Hydrating Foods: Watermelon, cucumber, ash gourd.</p>
                    </div>
                `
            },
            {
                condition: detectedRegions.nose > 50,
                suggestion: `
                    <div class="diet-card">
                        <h3>Heart Diet Suggestions</h3>
                        <p>Heart-Friendly Foods: Pomegranate, beetroot, almonds, walnuts.</p>
                        <p>Cooling Herbs: Arjuna, Brahmi, Tulsi tea.</p>
                        <p>Healthy Fats: Cow ghee, sesame seeds, flaxseeds.</p>
                    </div>
                `
            },
            {
                condition: detectedRegions.chin > 50,
                suggestion: `
                    <div class="diet-card">
                        <h3>Liver Diet Suggestions</h3>
                        <p>Liver Cleansing Foods: Turmeric, bitter gourd, neem, amla.</p>
                        <p>Herbs: Kutki, Bhumiamalaki, Aloe Vera.</p>
                        <p>Foods to Support Detox: Green leafy vegetables, beetroot, radish.</p>
                    </div>
                `
            }
        ];

        extraSuggestions.forEach(({ condition, suggestion }) => {
            if (condition) {
                dietSuggestions.insertAdjacentHTML('beforeend', suggestion);
            }
        });
    }
});
