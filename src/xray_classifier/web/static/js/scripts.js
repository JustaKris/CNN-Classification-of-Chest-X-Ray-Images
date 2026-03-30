/* ── Drag-and-drop + browse ────────────────────────────── */
(function () {
    var dropZone = document.getElementById("drop-zone");
    var fileInput = document.getElementById("file-upload");
    var uploadBtn = document.getElementById("btn-upload");
    var fileLabel = document.getElementById("drop-zone-file");

    if (!dropZone || !fileInput) return;

    function showFileName() {
        if (fileInput.files && fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
            dropZone.classList.add("has-file");
            uploadBtn.disabled = false;
        }
    }

    // Drag events
    ["dragenter", "dragover"].forEach(function (evt) {
        dropZone.addEventListener(evt, function (e) {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(function (evt) {
        dropZone.addEventListener(evt, function (e) {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", function (e) {
        var files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showFileName();
        }
    });

    // Click anywhere on the zone to browse
    dropZone.addEventListener("click", function () {
        fileInput.click();
    });

    // When file is chosen via the browser dialog
    fileInput.addEventListener("change", showFileName);
})();

/* ── "Try an example" link ────────────────────────────── */
(function () {
    var link = document.getElementById("try-example");
    var urlInput = document.getElementById("image-url");
    if (!link || !urlInput) return;

    link.addEventListener("click", function (e) {
        e.preventDefault();
        urlInput.value = "https://www.e7health.com/files/blogs/chest-x-ray-29.jpg";
        urlInput.focus();
    });
})();

/* ── Loading spinner on form submit ───────────────────── */
(function () {
    var overlay = document.getElementById("loading-overlay");
    if (!overlay) return;

    document.querySelectorAll("form").forEach(function (form) {
        form.addEventListener("submit", function () {
            overlay.classList.add("active");
        });
    });
})();