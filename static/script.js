document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    var formData = new FormData();
    var fileInput = document.getElementById('image-upload');
    formData.append('file', fileInput.files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '';
        for (var region in data) {
            resultDiv.innerHTML += `<p>${region}: ${(data[region] * 100).toFixed(2)}%</p>`;
        }
        
        var uploadedImage = document.getElementById('uploaded-image');
        uploadedImage.src = URL.createObjectURL(fileInput.files[0]);
    })
    .catch(error => console.error('Error:', error));
});
