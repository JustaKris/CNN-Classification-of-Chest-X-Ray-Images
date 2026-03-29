/* ── Drag-and-drop ─────────────────────────────────────── */
(function () {
    var dropZone = document.getElementById("drop-zone");
    var fileInput = document.getElementById("file-upload");
    var uploadForm = document.getElementById("upload-form");

    if (!dropZone || !fileInput) return;

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
            if (uploadForm) uploadForm.submit();
        }
    });

    dropZone.addEventListener("click", function () {
        fileInput.click();
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