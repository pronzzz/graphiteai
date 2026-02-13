document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-upload');
    const previewContainer = document.getElementById('preview-container');
    const originalImage = document.getElementById('original-image');
    const sketchImage = document.getElementById('sketch-image');
    const loadingSpinner = document.getElementById('loading');
    const actions = document.getElementById('actions');
    const downloadBtn = document.getElementById('download-btn');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and Drop
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
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'flex';
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        // Show loading
        loadingSpinner.style.display = 'block';
        sketchImage.style.display = 'none';
        actions.style.display = 'none';

        try {
            const response = await fetch('http://localhost:5001/sketch', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to generate sketch');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            sketchImage.src = url;
            sketchImage.style.display = 'block';
            actions.style.display = 'block';

            // Set up download
            downloadBtn.onclick = () => {
                const a = document.createElement('a');
                a.href = url;
                a.download = 'sketch.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };

        } catch (error) {
            console.error(error);
            alert('Error generating sketch: ' + error.message);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    }

    resetBtn.addEventListener('click', () => {
        dropZone.style.display = 'block'; // Was flex in CSS? check styles.css. styles.css didn't specify display for upload-section, so block is default.
        previewContainer.style.display = 'none';
        actions.style.display = 'none';
        fileInput.value = '';
        originalImage.src = '';
        sketchImage.src = '';
    });
});
