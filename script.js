// CIFAR-10 Class Names
const CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

let model;

// Enhanced model loader with absolute paths
async function loadModel() {
    try {
        showMessage('Loading model...', 'loading');
        
        // Use raw GitHub content URL
        const baseUrl = 'https://raw.githubusercontent.com/abhinav00215-hash/mtech/main/';
        
        // Load model with explicit weight path configuration
        model = await tf.loadLayersModel(
            tf.io.http(baseUrl + 'model.json', {
                weightPathPrefix: baseUrl,
                requestInit: { cache: 'no-store' } // Prevent caching issues
            }
        );
        
        showMessage('Model loaded successfully!', 'success');
        return true;
    } catch (err) {
        console.error('Model loading failed:', err);
        showMessage(`Failed to load model: ${err.message}`, 'error');
        return false;
    }
}

// Process image with better error handling
async function processImage(file) {
    if (!model) {
        const loaded = await loadModel();
        if (!loaded) return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.src = e.target.result;
        
        img.onload = async () => {
            document.getElementById('preview').innerHTML = '';
            document.getElementById('preview').appendChild(img);
            
            try {
                const predictions = await classifyImage(img);
                displayResults(predictions);
            } catch (err) {
                console.error('Classification error:', err);
                showMessage('Classification failed. Please try another image.', 'error');
            }
        };
    };
    reader.readAsDataURL(file);
}

// Classification function
async function classifyImage(img) {
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(img)
            .resizeNearestNeighbor([32, 32])
            .toFloat()
            .div(255.0)
            .expandDims();
    });

    const predictions = await model.predict(tensor).data();
    tensor.dispose();
    
    return Array.from(predictions)
        .map((p, i) => ({
            className: CLASS_NAMES[i],
            probability: p
        }))
        .sort((a, b) => b.probability - a.probability);
}

// Display results
function displayResults(predictions) {
    let html = '<div class="results">';
    html += `<h3>Top Prediction: ${predictions[0].className}</h3>`;
    html += `<p>Confidence: ${(predictions[0].probability * 100).toFixed(1)}%</p>`;
    
    html += '<h4>Other possibilities:</h4><ul>';
    predictions.slice(1, 4).forEach(p => {
        html += `<li>${p.className} (${(p.probability * 100).toFixed(1)}%)</li>`;
    });
    html += '</ul></div>';
    
    document.getElementById('prediction').innerHTML = html;
}

// Helper function
function showMessage(msg, type = 'info') {
    document.getElementById('prediction').innerHTML = 
        `<p class="${type}">${msg}</p>`;
}

// Initialize
window.onload = async () => {
    // Set up event listeners
    document.getElementById('fileInput').addEventListener('change', (e) => {
        if (e.target.files[0]) processImage(e.target.files[0]);
    });
    
    // Drag and drop
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', (e) => e.preventDefault());
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files[0]) processImage(e.dataTransfer.files[0]);
    });
    dropZone.addEventListener('click', () => document.getElementById('fileInput').click());

    // Load model
    await loadModel();
};
