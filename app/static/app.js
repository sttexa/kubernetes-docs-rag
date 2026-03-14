const form = document.getElementById("ask-form");
const result = document.getElementById("result");
const routeEl = document.getElementById("route");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("question").value.trim();
  if (!question) {
    return;
  }

  routeEl.textContent = "";
  answerEl.textContent = "Loading...";
  sourcesEl.innerHTML = "";
  result.classList.remove("hidden");

  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const payload = await response.json();
  if (!response.ok) {
    routeEl.textContent = "error";
    answerEl.textContent = payload.detail || "Request failed";
    return;
  }

  routeEl.textContent = payload.route;
  answerEl.textContent = payload.answer;
  for (const source of payload.sources) {
    const item = document.createElement("li");
    item.innerHTML = `<a href="${source.url}" target="_blank" rel="noreferrer">${source.title}</a> (${source.doc_type})<br>${source.excerpt}`;
    sourcesEl.appendChild(item);
  }
});

