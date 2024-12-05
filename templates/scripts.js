document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const videoInput = document.getElementById("videoInput");
    const file = videoInput.files[0];

    if (!file) {
        alert("Please select a video file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("video", file);

    try {
        const response = await fetch("/upload_video", {
            method: "POST",
            body: formData,
        });

        const statusDiv = document.getElementById("uploadStatus");
        if (response.ok) {
            const result = await response.json();
            statusDiv.innerHTML = `<p style="color: green;">${result.message}</p>`;
        } else {
            statusDiv.innerHTML = `<p style="color: red;">Error: ${response.statusText}</p>`;
        }
    } catch (error) {
        alert("An error occurred while uploading the video. Please try again.");
        console.error(error);
    }
});

document.getElementById("videoInput").addEventListener("change", function () {
    const allowedTypes = ["video/mp4", "video/webm", "video/ogg"];
    const maxFileSizeMB = 100;
    const file = this.files[0];

    if (file) {
        if (!allowedTypes.includes(file.type)) {
            alert("Invalid file type. Please upload a video file (MP4, WEBM, or OGG).");
            this.value = ""; 
            return;
        }

        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > maxFileSizeMB) {
            alert(`File size exceeds ${maxFileSizeMB} MB. Please upload a smaller file.`);
            this.value = ""; 
        }
    }
});

document.querySelectorAll('.faq .question').forEach((question) => {
    question.addEventListener('click', () => {
        const faq = question.parentElement;
        faq.classList.toggle('active');
    });
});