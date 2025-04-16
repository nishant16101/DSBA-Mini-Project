document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const uploadedImage = document.getElementById('uploaded-image');
    const resultContainer = document.getElementById('result');
    const proceedBtn = document.getElementById('proceed-btn');
    
    // Create custom file upload component
    if (uploadForm) {
        const fileLabel = document.createElement('label');
        fileLabel.className = 'file-upload-label';
        fileLabel.innerHTML = '<div class="upload-icon"><svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#4a90e2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg></div><span>Drag & drop an image or click to browse</span>';
        fileLabel.setAttribute('for', 'image-upload');
        
        // Insert the label before the existing button
        if (uploadForm.querySelector('.upload-btn')) {
            uploadForm.insertBefore(fileLabel, uploadForm.querySelector('.upload-btn'));
        } else {
            uploadForm.appendChild(fileLabel);
        }
    }
    
    // Handle file upload
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('image-upload');
            const file = fileInput.files[0];

            if (!file) {
                showNotification('Please upload an image first.', 'error');
                return;
            }

            // Show loading animation
            resultContainer.innerHTML = '<div class="loading"><div></div><div></div><div></div></div><p>Analyzing image...</p>';
            
            // Display the uploaded image
            const reader = new FileReader();
            
            reader.onload = function(event) {
                uploadedImage.src = event.target.result;
                uploadedImage.style.display = 'block';
                
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
                    // Convert percentage values to decimal for visualization
                    const normalizedData = {};
                    for (const key in data) {
                        normalizedData[key] = data[key] / 100;
                    }
                    
                    // Store detected regions in local storage
                    localStorage.setItem('detectedRegions', JSON.stringify(normalizedData));
                    
                    // Display results with face mapping visual
                    let resultHTML = '<h2>Analysis Results</h2>';
                    resultHTML += '<div class="face-mapping">';
                    resultHTML += '<img src="/static/images/face-outline.svg" alt="Face outline" class="face-outline">';
                    
                    // Add detected regions with appropriate opacity based on detection score
                    Object.keys(normalizedData).forEach(region => {
                        const score = normalizedData[region];
                        const opacity = Math.max(0.2, score);
                        
                        // Different classes based on region
                        let regionClass = '';
                        let regionPosition = '';
                        
                        if (region === 'forehead') {
                            regionClass = 'forehead-region';
                        } else if (region === 'cheeks') {
                            // For cheeks, create two regions
                            regionClass = 'cheeks-region';
                            regionPosition = '<div class="face-region cheeks-region left-cheek" style="opacity: ' + opacity + ';"></div>' +
                                           '<div class="face-region cheeks-region right-cheek" style="opacity: ' + opacity + ';"></div>';
                        } else if (region === 'nose') {
                            regionClass = 'nose-region';
                        } else if (region === 'chin') {
                            regionClass = 'chin-region';
                        }
                        
                        if (region !== 'cheeks') {
                            resultHTML += '<div class="face-region ' + regionClass + '" style="opacity: ' + opacity + ';"></div>';
                        } else {
                            resultHTML += regionPosition;
                        }
                    });
                    
                    resultHTML += '</div>';
                    
                    // Add text results
                    resultHTML += '<div class="detection-results">';
                    resultHTML += '<h3>Acne Detection Results:</h3>';
                    resultHTML += '<ul>';
                    
                    let hasSignificantAcne = false;
                    
                    Object.keys(data).forEach(region => {
                        const percentage = data[region];
                        let severity = "Low";
                        
                        if (percentage > 70) {
                            severity = "High";
                            hasSignificantAcne = true;
                        } else if (percentage > 40) {
                            severity = "Moderate";
                            hasSignificantAcne = true;
                        }
                        
                        const capitalized = region.charAt(0).toUpperCase() + region.slice(1);
                        resultHTML += '<li>' + capitalized + ': <span class="severity-' + severity.toLowerCase() + '">' + percentage + '%</span></li>';
                    });
                    
                    resultHTML += '</ul></div>';
                    
                    // Display result
                    resultContainer.innerHTML = resultHTML;
                    
                    // Show proceed button if acne is detected
                    if (hasSignificantAcne) {
                        proceedBtn.style.display = 'block';
                        showNotification('Acne detected! Click "Proceed" to see diet suggestions.', 'success');
                    } else {
                        showNotification('No significant acne detected.', 'info');
                    }
                    
                    // Add animation to the results
                    const faceRegions = document.querySelectorAll('.face-region');
                    faceRegions.forEach((region, index) => {
                        setTimeout(() => {
                            region.style.animation = 'fadeIn 0.5s forwards';
                        }, index * 200);
                    });
                })
                .catch(error => {
                    console.error('Error:', error);
                    resultContainer.innerHTML = '<p class="error-message">Error processing image. Please try again.</p>';
                    showNotification('Error processing image', 'error');
                });
            };
            
            reader.readAsDataURL(file);
        });
    }
    
    // Update button text when file is selected
    if (imageUpload) {
        imageUpload.addEventListener('change', function() {
            const uploadBtn = document.querySelector('.upload-btn');
            if (this.files[0] && uploadBtn) {
                uploadBtn.textContent = 'Analyze Image';
                uploadBtn.classList.add('ready');
            } else if (uploadBtn) {
                uploadBtn.textContent = 'Choose Image';
                uploadBtn.classList.remove('ready');
            }
        });
    }
    
    // Function to show diet suggestions page
    window.showDiet = function() {
        window.location.href = "/suggestions";  // Using direct URL as in original code
    };
    
    // Load diet suggestions on the diet page
    if (window.location.pathname.includes('suggestions')) {
        const dietSuggestions = document.getElementById('diet-suggestions');
        if (dietSuggestions) {
            dietSuggestions.innerHTML = '';
            
            const detectedRegions = JSON.parse(localStorage.getItem('detectedRegions')) || {};
            
            if (!detectedRegions || Object.keys(detectedRegions).length === 0) {
                dietSuggestions.innerHTML = `<p class="no-data">No acne detected or data unavailable.</p>`;
                return;
            }
            
            // Add introduction text
            const introText = document.createElement('div');
            introText.className = 'intro-text';
            introText.innerHTML = '<p>Based on your skin analysis, we\'ve created personalized diet suggestions to help improve your skin condition. Remember that diet is just one factor in managing acne - proper skincare, hydration, and stress management are also important.</p>';
            dietSuggestions.appendChild(introText);
            
            // Map regions to diet suggestions using the original suggestions
            const regionSuggestions = {
                forehead: `
                    <div class="diet-card">
                        <h3>Forehead Diet Suggestions</h3>
                        <p class="region-description">Forehead acne is often linked to digestive issues and liver function.</p>
                        <p><strong>Cooling Foods:</strong> Cucumber, coriander, fennel, coconut water.</p>
                        <p><strong>Herbs:</strong> Triphala, licorice, aloe vera juice.</p>
                        <p><strong>Easily Digestible Foods:</strong> Moong dal, rice, ghee, warm soups.</p>
                        <p><strong>Foods to Avoid:</strong> Fried foods, spicy foods, alcohol, excessive caffeine.</p>
                    </div>
                `,
                cheeks: `
                    <div class="diet-card">
                        <h3>Cheeks Diet Suggestions</h3>
                        <p class="region-description">Cheek acne may be related to respiratory system and kidney function.</p>
                        <p><strong>Diuretic Foods:</strong> Barley water, cranberry juice, coconut water.</p>
                        <p><strong>Herbs:</strong> Punarnava, gokshura, coriander seeds tea.</p>
                        <p><strong>Hydrating Foods:</strong> Watermelon, cucumber, ash gourd.</p>
                        <p><strong>Foods to Avoid:</strong> Excess salt, processed foods, dehydrating beverages.</p>
                    </div>
                `,
                nose: `
                    <div class="diet-card">
                        <h3>Nose Diet Suggestions</h3>
                        <p class="region-description">Nose acne often relates to heart and circulation issues.</p>
                        <p><strong>Heart-Friendly Foods:</strong> Pomegranate, beetroot, almonds, walnuts.</p>
                        <p><strong>Cooling Herbs:</strong> Arjuna, Brahmi, Tulsi tea.</p>
                        <p><strong>Healthy Fats:</strong> Cow ghee, sesame seeds, flaxseeds.</p>
                        <p><strong>Foods to Avoid:</strong> Excessive fatty foods, too much salt, alcohol.</p>
                    </div>
                `,
                chin: `
                    <div class="diet-card">
                        <h3>Chin Diet Suggestions</h3>
                        <p class="region-description">Chin acne is commonly connected to hormonal imbalances.</p>
                        <p><strong>Liver Cleansing Foods:</strong> Turmeric, bitter gourd, neem, amla.</p>
                        <p><strong>Herbs:</strong> Kutki, Bhumiamalaki, Aloe Vera.</p>
                        <p><strong>Foods to Support Detox:</strong> Green leafy vegetables, beetroot, radish.</p>
                        <p><strong>Foods to Avoid:</strong> Dairy products, sugar, refined carbohydrates.</p>
                    </div>
                `
            };
            
            // Display suggestions for regions with significant acne (threshold at 0.5 or 50%)
            Object.keys(regionSuggestions).forEach((region, index) => {
                const threshold = region in detectedRegions ? 0.5 : 50/100; // Handle both normalized and percentage values
                if (detectedRegions[region] > threshold) {
                    const card = document.createElement('div');
                    card.className = 'diet-card';
                    card.style.animationDelay = (index * 0.1) + 's';
                    card.innerHTML = regionSuggestions[region];
                    dietSuggestions.appendChild(card);
                }
            });
            
            // Add the additional holistic suggestions from original code
            const extraSuggestions = [
                {
                    condition: detectedRegions.forehead > 0.5,
                    suggestion: `
                        <div class="diet-card">
                            <h3>Small Intestine Diet Suggestions</h3>
                            <p class="region-description">Forehead acne can indicate digestive system imbalances.</p>
                            <p><strong>Cooling Foods:</strong> Cucumber, coriander, fennel, coconut water.</p>
                            <p><strong>Herbs:</strong> Triphala, licorice, aloe vera juice.</p>
                            <p><strong>Easily Digestible Foods:</strong> Moong dal, rice, ghee, warm soups.</p>
                            <p><strong>Foods to Avoid:</strong> Heavy, fried foods, processed foods, excessive oil.</p>
                        </div>
                    `
                },
                {
                    condition: detectedRegions.cheeks > 0.5,
                    suggestion: `
                        <div class="diet-card">
                            <h3>Bladder Diet Suggestions</h3>
                            <p class="region-description">Cheek acne can indicate bladder and kidney imbalances.</p>
                            <p><strong>Diuretic Foods:</strong> Barley water, cranberry juice, coconut water.</p>
                            <p><strong>Herbs:</strong> Punarnava, gokshura, coriander seeds tea.</p>
                            <p><strong>Hydrating Foods:</strong> Watermelon, cucumber, ash gourd.</p>
                            <p><strong>Foods to Avoid:</strong> Excess salt, caffeine, alcohol, spicy foods.</p>
                        </div>
                    `
                },
                {
                    condition: detectedRegions.nose > 0.5,
                    suggestion: `
                        <div class="diet-card">
                            <h3>Heart Diet Suggestions</h3>
                            <p class="region-description">Nose acne can indicate heart and circulation imbalances.</p>
                            <p><strong>Heart-Friendly Foods:</strong> Pomegranate, beetroot, almonds, walnuts.</p>
                            <p><strong>Cooling Herbs:</strong> Arjuna, Brahmi, Tulsi tea.</p>
                            <p><strong>Healthy Fats:</strong> Cow ghee, sesame seeds, flaxseeds.</p>
                            <p><strong>Foods to Avoid:</strong> Excessive fatty foods, fried foods, salt.</p>
                        </div>
                    `
                },
                {
                    condition: detectedRegions.chin > 0.5,
                    suggestion: `
                        <div class="diet-card">
                            <h3>Liver Diet Suggestions</h3>
                            <p class="region-description">Chin acne can indicate liver imbalances and hormonal issues.</p>
                            <p><strong>Liver Cleansing Foods:</strong> Turmeric, bitter gourd, neem, amla.</p>
                            <p><strong>Herbs:</strong> Kutki, Bhumiamalaki, Aloe Vera.</p>
                            <p><strong>Foods to Support Detox:</strong> Green leafy vegetables, beetroot, radish.</p>
                            <p><strong>Foods to Avoid:</strong> Alcohol, fatty foods, excessive spices.</p>
                        </div>
                    `
                }
            ];

            extraSuggestions.forEach(({ condition, suggestion }, index) => {
                if (condition) {
                    const extraCard = document.createElement('div');
                    extraCard.className = 'diet-card holistic';
                    extraCard.style.animationDelay = (index * 0.1 + 0.4) + 's';
                    extraCard.innerHTML = suggestion;
                    dietSuggestions.appendChild(extraCard);
                }
            });
            
            // Add general skin health tips
            const generalCard = document.createElement('div');
            generalCard.className = 'diet-card general-tips';
            generalCard.style.animationDelay = '0.8s';
            generalCard.innerHTML = `
                <h3>General Skin Health Tips</h3>
                <p>These suggestions benefit all skin types:</p>
                <ul>
                    <li>Stay hydrated with at least 8 glasses of water daily</li>
                    <li>Include antioxidant-rich foods like berries, green tea, and colorful vegetables</li>
                    <li>Consume omega-3 fatty acids found in flaxseeds, walnuts, and fatty fish</li>
                    <li>Reduce sugar intake and processed foods</li>
                    <li>Include probiotics for gut health (yogurt, kefir, fermented foods)</li>
                </ul>
            `;
            dietSuggestions.appendChild(generalCard);
        }
    }
    
    // Notification function
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = 'notification ' + type;
        notification.innerHTML = message;
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateY(0)';
            notification.style.opacity = '1';
        }, 10);
        
        // Animate out and remove
        setTimeout(() => {
            notification.style.transform = 'translateY(-20px)';
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 4000);
    }
    
    // Add notification styles if not already in CSS
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.innerHTML = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                background-color: #f8f9fa;
                border-left: 4px solid #4a90e2;
                border-radius: 4px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                z-index: 1000;
                transform: translateY(-20px);
                opacity: 0;
                transition: all 0.3s ease;
            }
            .notification.success {
                border-left-color: #28a745;
            }
            .notification.error {
                border-left-color: #dc3545;
            }
            .notification.info {
                border-left-color: #17a2b8;
            }
        `;
        document.head.appendChild(style);
    }
});