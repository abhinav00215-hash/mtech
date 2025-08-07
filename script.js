// CIFAR-10 Class Names
const CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

let model;

async function loadModel() {
    try {
        showMessage('Loading model...', 'loading');
        
        // Use GitHub Pages URL for all files
        const baseUrl = 'https://abhinav00215-hash.github.io/mtech/';
        
        // Load model with standard loader
        model = await tf.loadLayersModel(baseUrl + 'model.json');
        
        showMessage('Model loaded successfully!', 'success');
        return true;
    } catch (err) {
        console.error('Model loading failed:', err);
        
        // Fallback to raw GitHub URLs if needed
        try {
            showMessage('Trying alternative loading method...', 'loading');
            const rawBaseUrl = 'https://raw.githubusercontent.com/abhinav00215-hash/mtech/main/';
            model = await tf.loadLayersModel(rawBaseUrl + 'model.json');
            showMessage('Model loaded successfully!', 'success');
            return true;
        } catch (fallbackErr) {
            console.error('Fallback loading failed:', fallbackErr);
            showMessage('Failed to load model. Please check console for details.', 'error');
            return false;
        }
    }
}

// Rest of your existing functions remain the same...
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

// ... (keep all other functions exactly as they were)
