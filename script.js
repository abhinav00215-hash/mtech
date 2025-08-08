// CIFAR-10 Class Names
const CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

let model;

// Show messages in the prediction div
function showMessage(message, type) {
    const predictionDiv = document.getElementById('prediction');
    predictionDiv.innerHTML = `<span class="${type}">${message}</span>`;
}

// Load model from GitHub Pages
async function loadModel() {
    try {
        showMessage('Loading model...', 'loading');
        
        // If model.json & .bin files are in same folder as index.html
        const baseUrl = 'https://abhinav00215-hash.github.io/mtech/';

        model = await tf.loadLayersModel(baseUrl + 'model.json');
        
        showMessage('Model loaded successfully!', 'success');
        return true;
    } catch (err) {
        console.error('Model loading failed:', err);
        showMessage('Failed to load model.', 'error');
        return false;
    }
}

// Classify the image
async function classifyImage(img) {
    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([32, 32])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();

    const predictions = await model.predict(tensor).data();
    const results = Array.from(predictions)
        .map((p, i) => ({ className: CLASS_NAMES[i], probability: p }))
        .sort((a, b) => b.probability - a.probability);

    return results;
}

// Process uploaded image
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
    };
    reader.readAsDataURL(file);
}

// Display classification results
function displayResults(predictions) {
    const predictionDiv = document.getElementById('prediction');
    predictionDiv.innerHTML = predictions.slice(0, 3).map(p => 
        `<div>
            <span class="class-name">${p.className}</span> - 
            <span class="confidence">${(p.probability * 100).toFixed(2)}%</span>
        </div>`
    ).join('');
}

// Event listeners
document.getElementById('fileInput').addEventListener('change', function (e) {
    processImage(e.target.files[0]);
});

document.getElementById('predictBtn').addEventListener('click', function () {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        showMessage('Please select an image first.', 'error');
    } else {
        processImage(fileInput.files[0]);
    }
});

// Drag & Drop Support
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('click', () => document.getElementById('fileInput').click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#4CAF50';
});
dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#ccc';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#ccc';
    if (e.dataTransfer.files.length > 0) {
        processImage(e.dataTransfer.files[0]);
    }
});
