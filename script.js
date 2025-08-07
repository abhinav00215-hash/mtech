// CIFAR-10 Class Names
const CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

let model;

async function loadModel() {
    try {
        showMessage('Loading model...', 'loading');
        
        // Use GitHub Pages URL for model.json and direct GitHub URLs for weights
        const modelUrl = 'https://abhinav00215-hash.github.io/mtech/model.json';
        const weightsBaseUrl = 'https://raw.githubusercontent.com/abhinav00215-hash/mtech/main/';
        
        // Custom loader to handle GitHub's raw content URLs for weights
        const customLoader = {
            load: async () => {
                const modelTopology = await fetch(modelUrl).then(res => res.json());
                const weightManifest = modelTopology.weightsManifest[0];
                
                // Map weights to raw GitHub URLs
                weightManifest.paths = weightManifest.paths.map(path => 
                    weightsBaseUrl + path
                );
                
                return {
                    modelTopology,
                    weightSpecs: weightManifest.weights,
                    weightData: await fetchWeights(weightManifest.paths)
                };
            }
        };
        
        model = await tf.loadLayersModel(customLoader);
        showMessage('Model loaded successfully! Upload an image to classify.', 'success');
        return true;
    } catch (err) {
        console.error('Model loading failed:', err);
        showMessage(`Failed to load model: ${err.message}`, 'error');
        return false;
    }
}

// Helper to fetch weights from multiple URLs
async function fetchWeights(weightUrls) {
    const weightPromises = weightUrls.map(url => 
        fetch(url).then(res => res.arrayBuffer())
    );
    const weightBuffers = await Promise.all(weightPromises);
    
    // Concatenate all ArrayBuffers
    const totalLength = weightBuffers.reduce((sum, buf) => sum + buf.byteLength, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    
    weightBuffers.forEach(buf => {
        result.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
    });
    
    return result.buffer;
}

// Rest of your existing code remains the same...
async function processImage(file) {
    if (!file.type.match('image.*')) {
        showMessage('Please upload an image file (JPEG, PNG, etc.)', 'error');
        return;
    }

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
            showMessage('Classifying image...', 'loading');
            
            try {
                const predictions = await classifyImage(img);
                displayResults(predictions);
            } catch (err) {
                console.error('Classification error:', err);
                showMessage('Classification failed. Please try another image.', 'error');
            }
        };
        
        img.onerror = () => {
            showMessage('Error loading image', 'error');
        };
    };
    
    reader.onerror = () => {
        showMessage('Error reading file', 'error');
    };
    
    reader.readAsDataURL(file);
}

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

function displayResults(predictions) {
    if (!predictions || predictions.length === 0) {
        showMessage('No predictions were generated', 'error');
        return;
    }

    let html = '<div class="results">';
    html += `<h3>Prediction: <span class="class-name">${predictions[0].className}</span></h3>`;
    html += `<p>Confidence: <span class="confidence">${(predictions[0].probability * 100).toFixed(1)}%</span></p>`;
    
    if (predictions.length > 1) {
        html += '<h4>Other possibilities:</h4><ul>';
        predictions.slice(1, 4).forEach(p => {
            if (p.probability > 0.1) {
                html += `<li>${p.className} <span class="confidence">(${(p.probability * 100).toFixed(1)}%)</span></li>`;
            }
        });
        html += '</ul>';
    }
    
    html += '</div>';
    document.getElementById('prediction').innerHTML = html;
}

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
        if (e.dataTransfer.files[0]) processImage(e.dataTransfer.files[0]);
    });
    
    // Click to browse
    dropZone.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });

    // Load model
    await loadModel();
};
