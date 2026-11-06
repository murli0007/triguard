const result = document.getElementById("result");
function render(data) {
  if (!data) { result.className = ""; result.textContent = "No URL result is available for this tab yet."; return; }
  if (data.status === "checking") {
    result.className = ""; result.textContent = `Checking ${data.url}…`; return;
  }
  if (data.status === "error" || data?.error) {
    result.className = "error"; result.textContent = `Backend unavailable: ${data.error || "URL scan failed."}`; return;
  }
  if (data.status === "complete" && data.result) data = data.result;
  if (!data.prediction || typeof data.risk_score !== "number") {
    result.className = "error"; result.textContent = "The backend returned an unreadable URL result."; return;
  }
  result.className = data.prediction || "";
  result.replaceChildren();
  const title = document.createElement("strong"); title.textContent = `${data.prediction.toUpperCase()} · ${data.risk_score}/100`;
  const confidence = document.createElement("div"); confidence.textContent = `Model probability: ${data.model_probability == null ? "not available" : `${(data.model_probability*100).toFixed(1)}%${data.probability_calibrated ? " (calibrated)" : " (uncalibrated)"}`}`;
  result.append(title, confidence);
  if (data.model) {
    const model = document.createElement("div");
    model.className = "reason";
    model.textContent = `Model: ${data.model} · Source: ${data.model_source || data.training_source || "unknown"}`;
    result.append(model);
  }
  for (const reason of (data.reasons || []).slice(0,3)) { const line=document.createElement("p");line.className="reason";line.textContent=`• ${reason}`;result.append(line); }
}
function send(message) { return chrome.runtime.sendMessage(message); }
async function activeTab() { return (await chrome.tabs.query({active:true,currentWindow:true}))[0]; }
async function showLatestForActiveTab() {
  const tab = await activeTab();
  if (!/^https?:\/\//i.test(tab?.url || "")) {
    result.textContent = "Only HTTP and HTTPS pages are scanned. Browser-internal pages are ignored.";
    return;
  }
  const state = await send({type:"GET_TAB_URL_RESULT", tabId:tab.id});
  if (state?.url && state.url !== tab.url) {
    result.textContent = "Waiting for the current page URL scan…";
    return;
  }
  render(state);
}
document.getElementById("checkUrl").addEventListener("click", async () => {
  const tab=await activeTab(); if(!/^https?:\/\//i.test(tab?.url||"")){result.textContent="Only HTTP and HTTPS pages can be checked.";return;}
  result.textContent="Checking current URL…";
  try { render(await send({type:"CHECK_URL",url:tab.url,tabId:tab.id})); }
  catch (error) { render({status:"error",error:error.message}); }
});
document.getElementById("checkEmail").addEventListener("click", async () => {
  const tab=await activeTab(); if(!tab?.id)return;
  result.textContent="Reading visible Gmail message…";
  chrome.tabs.sendMessage(tab.id,{type:"GET_VISIBLE_EMAIL"},async mail=>{
    if(chrome.runtime.lastError||!mail?.body){result.className="error";result.textContent="Open an email message in Gmail, then try again.";return;}
    result.textContent="Analyzing email text locally…";render(await send({type:"CHECK_EMAIL",subject:mail.subject,body:mail.body}));
  });
});
showLatestForActiveTab().catch(error => {
  result.className="error";
  result.textContent=`Could not load this tab's latest result: ${error.message}`;
});
