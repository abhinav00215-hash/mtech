// CIFAR-10 Class Names
const CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

let model;

// Load the TensorFlow.js model
async function loadModel() {
    try {
        document.getElementById('prediction').innerHTML = 
            '<p class="loading">Loading model...</p>';
        
        model = await tf.loadLayersModel('model.json');
        
        document.getElementById('prediction').innerHTML = 
            '<p>Model loaded! Upload an image to classify.</p>';
        console.log('Model loaded successfully');
    } catch (err) {
        console.error('Error loading model:', err);
        document.getElementById('prediction').innerHTML = 
            '<p class="error">Error loading model. Check console.</p>';
    }
}

// Handle image upload
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        processImage(file);
    }
});

// Handle drag and drop
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.match('image.*')) {
        processImage(file);
    }
});

// Click to browse
dropZone.addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// Process the uploaded image
async function processImage(file) {
    const preview = document.getElementById('preview');
    const predictionDiv = document.getElementById('prediction');
    
    // Clear previous results
    preview.innerHTML = '';
    predictionDiv.innerHTML = '<p class="loading">Processing image...</p>';
    
    // Display the image
    const reader = new FileReader();
    reader.onload = async function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.onload = () => {
            preview.appendChild(img);
            classifyImage(img).then(displayResults).catch(handleClassificationError);
        };
    };
    reader.readAsDataURL(file);
}

// Classify the image using the model
async function classifyImage(imgElement) {
    // Convert image to tensor
    const tensor = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([32, 32])  // CIFAR-10 input size
        .toFloat()
        .div(tf.scalar(255.0))  // Normalize to [0,1]
        .expandDims();
    
    // Predict
    const predictions = await model.predict(tensor).data();
    tensor.dispose();  // Clean up
    
    // Convert to readable format
    return Array.from(predictions)
        .map((p, i) => ({
            className: CLASS_NAMES[i],
            probability: p
        }))
        .sort((a, b) => b.probability - a.probability);
}

// Handle classification errors
function handleClassificationError(err) {
    console.error('Error classifying image:', err);
    document.getElementById('prediction').innerHTML = 
        '<p class="error">Error classifying image. Check console.</p>';
}

// Display the prediction results
function displayResults(predictions) {
    const predictionDiv = document.getElementById('prediction');
    let html = '<h3>Prediction Results:</h3>';
    
    if (!predictions || predictions.length === 0) {
        html += '<p class="error">No predictions were returned</p>';
        predictionDiv.innerHTML = html;
        return;
    }

    // Top prediction
    const top = predictions[0];
    html += `<p>Most likely: <strong>${top.className}</strong> ` +
            `(<span class="confidence">${(top.probability * 100).toFixed(1)}%</span> confidence)</p>`;
    
    // Top 3 predictions
    if (predictions.length > 1) {
        html += '<h4>Other possibilities:</h4><ul>';
        for (let i = 1; i < Math.min(3, predictions.length); i++) {
            const p = predictions[i];
            html += `<li>${p.className} ` +
                    `(<span class="confidence">${(p.probability * 100).toFixed(1)}%</span>)</li>`;
        }
        html += '</ul>';
    }
    
    predictionDiv.innerHTML = html;
}

// Initialize when page loads
window.onload = function() {
    loadModel();
};

