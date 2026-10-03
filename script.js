const message = document.getElementById("message");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultTitle = document.getElementById("resultTitle");
const resultMeta = document.getElementById("resultMeta");

document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    message.value = btn.dataset.example;
    message.focus();
  });
});

clearBtn.addEventListener("click", () => {
  message.value = "";
  result.style.display = "none";
});

analyzeBtn.addEventListener("click", async () => {
  const text = message.value.trim();

  if (!text) {
    message.focus();
    message.style.borderColor = "rgba(255,85,119,.7)";
    setTimeout(() => message.style.borderColor = "", 700);
    return;
  }

  result.style.display = "none";
  loading.style.display = "flex";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const spam = data.prediction === 1;
    result.className = `result ${spam ? "spam" : "ham"}`;
    resultIcon.textContent = spam ? "!" : "✓";
    resultLabel.textContent = "CLASSIFICATION";
    resultTitle.textContent = spam ? "SPAM DETECTED" : "NOT SPAM";
    resultMeta.textContent =
      `${data.confidence}% model confidence • TF-IDF + Multinomial Naive Bayes`;

    result.style.display = "flex";
  } catch (error) {
    result.className = "result spam";
    resultIcon.textContent = "×";
    resultLabel.textContent = "ERROR";
    resultTitle.textContent = "Could not classify";
    resultMeta.textContent = error.message;
    result.style.display = "flex";
  } finally {
    loading.style.display = "none";
    analyzeBtn.disabled = false;
  }
});
