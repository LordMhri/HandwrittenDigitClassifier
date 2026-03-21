/**
 * MNIST Hand-Written Digit Classifier - Web Interface
 * Logic: Canvas Drawing -> 28x28 Grayscale Downscaling -> NN Forward Pass
 */

class DigitClassifier {
    constructor() {
        this.canvas = document.getElementById('digit-canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.isDrawing = false;
        this.weights = null;
        
        this.initCanvas();
        this.initChart();
        this.loadModel();
        this.bindEvents();
    }

    initCanvas() {
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.lineWidth = 16;
        this.ctx.lineCap = "round";
        this.ctx.strokeStyle = "white";
    }

    initChart() {
        const chart = document.getElementById('probability-chart');
        chart.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-label">${i}</div>
                <div class="prob-bar-bg"><div id="bar-${i}" class="prob-bar-fill" style="width: 0%"></div></div>
                <div id="val-${i}" class="prob-value">0%</div>
            `;
            chart.appendChild(row);
        }
    }

    async loadModel() {
        try {
            // Loading from the parent directory where C engine saves it
            const response = await fetch('../src/model_weights.json');
            if (!response.ok) throw new Error('Could not load model_weights.json');
            this.weights = await response.json();
            console.log('Model loaded successfully:', this.weights.layer_sizes);
        } catch (err) {
            console.error('Error loading model:', err);
            document.querySelector('header p').textContent = "Error: model_weights.json not found. Run the C trainer first!";
            document.querySelector('header p').style.color = "#f87171";
        }
    }

    bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseleave', () => this.stopDrawing());

        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        this.canvas.addEventListener('touchend', () => this.stopDrawing());

        document.getElementById('clear-btn').addEventListener('click', () => {
            this.initCanvas();
            this.resetResults();
        });

        document.getElementById('predict-btn').addEventListener('click', () => this.performPrediction());
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.ctx.beginPath();
        const { x, y } = this.getMousePos(e);
        this.ctx.moveTo(x, y);
    }

    draw(e) {
        if (!this.isDrawing) return;
        const { x, y } = this.getMousePos(e);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.performPrediction(); // Predict immediately on release
        }
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        // Scale coordinates to canvas resolution in case CSS resizes it
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    resetResults() {
        document.getElementById('predicted-digit').textContent = '-';
        for (let i = 0; i < 10; i++) {
            document.getElementById(`bar-${i}`).style.width = '0%';
            document.getElementById(`val-${i}`).textContent = '0%';
        }
    }

    /**
     * Compute bounding box and center-of-mass of drawn content,
     * then produce a centered 20x20-in-28x28 image (matching MNIST preprocessing).
     */
    getGrayscalePixels() {
        // Step 1: Get the full-resolution image data to find bounding box & center of mass
        const srcData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const srcW = this.canvas.width;
        const srcH = this.canvas.height;

        let minX = srcW, minY = srcH, maxX = 0, maxY = 0;
        let massX = 0, massY = 0, totalMass = 0;

        for (let y = 0; y < srcH; y++) {
            for (let x = 0; x < srcW; x++) {
                const idx = (y * srcW + x) * 4;
                // Luminance-weighted grayscale
                const gray = (srcData.data[idx] * 0.299 +
                              srcData.data[idx + 1] * 0.587 +
                              srcData.data[idx + 2] * 0.114) / 255.0;
                if (gray > 0.01) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    massX += x * gray;
                    massY += y * gray;
                    totalMass += gray;
                }
            }
        }

        // If canvas is empty, return zeros
        if (totalMass === 0) return new Float32Array(784);

        // Step 2: Crop to bounding box and scale into a 20x20 region (MNIST convention)
        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;

        // Fit the bounding box into 20x20 while preserving aspect ratio
        const scale = 20 / Math.max(bw, bh);
        const scaledW = Math.round(bw * scale);
        const scaledH = Math.round(bh * scale);

        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = scaledW;
        tmpCanvas.height = scaledH;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.imageSmoothingEnabled = true;
        tmpCtx.imageSmoothingQuality = 'high';
        tmpCtx.drawImage(this.canvas, minX, minY, bw, bh, 0, 0, scaledW, scaledH);

        // Step 3: Compute center of mass of the scaled image
        const tmpData = tmpCtx.getImageData(0, 0, scaledW, scaledH);
        let cmX = 0, cmY = 0, cmTotal = 0;
        for (let y = 0; y < scaledH; y++) {
            for (let x = 0; x < scaledW; x++) {
                const idx = (y * scaledW + x) * 4;
                const gray = (tmpData.data[idx] * 0.299 +
                              tmpData.data[idx + 1] * 0.587 +
                              tmpData.data[idx + 2] * 0.114) / 255.0;
                cmX += x * gray;
                cmY += y * gray;
                cmTotal += gray;
            }
        }
        cmX /= cmTotal;
        cmY /= cmTotal;

        // Step 4: Place the scaled image on a 28x28 canvas, centered by center-of-mass
        const outCanvas = document.createElement('canvas');
        outCanvas.width = 28;
        outCanvas.height = 28;
        const outCtx = outCanvas.getContext('2d');
        outCtx.fillStyle = 'black';
        outCtx.fillRect(0, 0, 28, 28);
        outCtx.imageSmoothingEnabled = true;
        outCtx.imageSmoothingQuality = 'high';

        // Offset so that the center of mass lands on pixel (14, 14)
        const dx = Math.round(14 - cmX);
        const dy = Math.round(14 - cmY);
        outCtx.drawImage(tmpCanvas, dx, dy);

        // Step 5: Extract grayscale pixel values
        const imageData = outCtx.getImageData(0, 0, 28, 28);
        const pixels = new Float32Array(784);
        for (let i = 0; i < 784; i++) {
            const idx = i * 4;
            pixels[i] = (imageData.data[idx] * 0.299 +
                         imageData.data[idx + 1] * 0.587 +
                         imageData.data[idx + 2] * 0.114) / 255.0;
        }
        return pixels;
    }

    /**
     * Neural Network Forward Pass
     */
    performPrediction() {
        if (!this.weights) return;

        const input = this.getGrayscalePixels();
        let currentLayer = input;

        // Iterate through layer transitions
        for (let i = 0; i < this.weights.num_layers - 1; i++) {
            const nextLayerSize = this.weights.layer_sizes[i + 1];
            const currentLayerSize = this.weights.layer_sizes[i];
            const nextLayer = new Float32Array(nextLayerSize);

            const layerWeights = this.weights.weights[i];
            const layerBiases = this.weights.biases[i];

            // Matrix Multiply: target = source * weights
            for (let j = 0; j < nextLayerSize; j++) {
                let sum = 0;
                for (let k = 0; k < currentLayerSize; k++) {
                    sum += currentLayer[k] * layerWeights[k * nextLayerSize + j];
                }
                sum += layerBiases[j];

                // Activation
                if (i < this.weights.num_layers - 2) {
                    // ReLU for hidden layers
                    nextLayer[j] = Math.max(0, sum);
                } else {
                    // Final layer: Raw logits before softmax
                    nextLayer[j] = sum;
                }
            }
            currentLayer = nextLayer;
        }

        // Apply Softmax to output layer
        const probabilities = this.softmax(currentLayer);
        this.updateUI(probabilities);
    }

    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b);
        return exps.map(x => x / sumExps);
    }

    updateUI(probs) {
        let maxIdx = 0;
        let maxProb = 0;

        probs.forEach((p, i) => {
            const percent = (p * 100).toFixed(1);
            document.getElementById(`bar-${i}`).style.width = `${percent}%`;
            document.getElementById(`val-${i}`).textContent = `${percent}%`;
            
            if (p > maxProb) {
                maxProb = p;
                maxIdx = i;
            }
        });

        document.getElementById('predicted-digit').textContent = maxIdx;
    }
}

// Initialize the app
window.addEventListener('load', () => {
    new DigitClassifier();
});
