const API = "http://127.0.0.1:5050";
const lastNavigationUrlByTab = new Map();
const latestRequestByTab = new Map();
const inFlightByTabAndUrl = new Map();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "CHECK_URL") {
    const tabId = Number.isInteger(message.tabId) ? message.tabId : sender.tab?.id;
    if (!isWebUrl(message.url)) {
      sendResponse({error: "Only HTTP and HTTPS URLs can be checked."});
      return false;
    }
    scanUrl(message.url, tabId, "popup")
      .then(sendResponse)
      .catch(error => sendResponse({error: error.message}));
    return true;
  }
  if (message?.type === "GET_TAB_URL_RESULT") {
    if (!Number.isInteger(message.tabId)) {
      sendResponse({error: "No active tab is available."});
      return false;
    }
    getTabResult(message.tabId).then(sendResponse)
      .catch(error => sendResponse({error: error.message}));
    return true;
  }
  if (message?.type === "CHECK_EMAIL") {
    checkEmail(message.subject || "", message.body || "").then(sendResponse)
      .catch(error => sendResponse({error: error.message}));
    return true;
  }
  if (message?.type === "AUTO_CHECK_EMAIL") {
    checkEmail(message.subject || "", message.body || "")
      .then(result => {
        if (!result.error && result.risk_score >= 30 && sender.tab?.id) {
          chrome.tabs.sendMessage(sender.tab.id, {type: "EMAIL_WARNING", result}).catch(() => {});
        }
        sendResponse(result);
      }).catch(error => sendResponse({error: error.message}));
    return true;
  }
});

// Main-frame commits include the final URL after redirects. SPA URL changes are
// covered as well; duplicate events for the same URL are collapsed below.
function checkNavigation(details) {
  if (details.frameId !== 0 || !isWebUrl(details.url)) {
    console.info("[Sentinel URL] Ignored non-web or subframe navigation", details.url);
    return;
  }
  if (lastNavigationUrlByTab.get(details.tabId) === details.url) {
    console.info("[Sentinel URL] Skipping duplicate navigation", {tabId: details.tabId, url: details.url});
    return;
  }
  lastNavigationUrlByTab.set(details.tabId, details.url);
  scanUrl(details.url, details.tabId, "navigation").catch(error => {
    console.error("[Sentinel URL] Unexpected scan failure", {tabId: details.tabId, url: details.url, error});
  });
}
chrome.webNavigation.onCommitted.addListener(checkNavigation);
chrome.webNavigation.onHistoryStateUpdated.addListener(checkNavigation);
chrome.webNavigation.onReferenceFragmentUpdated.addListener(checkNavigation);
chrome.tabs.onRemoved.addListener(tabId => {
  lastNavigationUrlByTab.delete(tabId);
  latestRequestByTab.delete(tabId);
  chrome.storage.local.remove(resultKey(tabId)).catch(error => {
    console.warn("[Sentinel URL] Could not clear closed-tab result", error);
  });
});

async function scanUrl(url, tabId, source) {
  if (!isWebUrl(url)) throw new Error("Only HTTP and HTTPS URLs can be checked.");

  if (!Number.isInteger(tabId)) {
    console.info("[Sentinel URL] Scanning URL without tab storage", {source, url});
    return checkUrl(url);
  }

  const flightKey = `${tabId}\n${url}`;
  const existingFlight = inFlightByTabAndUrl.get(flightKey);
  if (existingFlight) {
    console.info("[Sentinel URL] Reusing in-flight request", {tabId, url, source});
    return existingFlight;
  }

  const requestId = (latestRequestByTab.get(tabId) || 0) + 1;
  latestRequestByTab.set(tabId, requestId);
  lastNavigationUrlByTab.set(tabId, url);
  await saveTabResult(tabId, {status: "checking", url, updatedAt: Date.now()});
  console.info("[Sentinel URL] Sending URL to backend", {tabId, source, url});

  const request = checkUrl(url).then(async result => {
    if (latestRequestByTab.get(tabId) === requestId) {
      await saveTabResult(tabId, {status: "complete", url, result, updatedAt: Date.now()});
      console.info("[Sentinel URL] Backend prediction received", {
        tabId, url, prediction: result.prediction, riskScore: result.risk_score,
        model: result.model, source: result.model_source || result.training_source
      });
      if (result.risk_score >= 30) {
        deliverWarning(tabId, {type: "URL_WARNING", result}).catch(() => {});
      }
    } else {
      console.info("[Sentinel URL] Ignoring stale backend response", {tabId, url});
    }
    return result;
  }).catch(async error => {
    if (latestRequestByTab.get(tabId) === requestId) {
      await saveTabResult(tabId, {status: "error", url, error: error.message, updatedAt: Date.now()});
    }
    console.error("[Sentinel URL] Backend request failed", {tabId, url, error: error.message});
    return {error: error.message, url};
  }).finally(() => {
    if (inFlightByTabAndUrl.get(flightKey) === request) inFlightByTabAndUrl.delete(flightKey);
  });

  inFlightByTabAndUrl.set(flightKey, request);
  return request;
}

async function deliverWarning(tabId, message, attempt = 0) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch (error) {
    if (attempt < 4) setTimeout(() => deliverWarning(tabId, message, attempt + 1), 500);
    else console.info("[Sentinel URL] No content script available for warning", {tabId});
  }
}

function isWebUrl(url) { return /^https?:\/\//i.test(url || ""); }
function resultKey(tabId) { return `urlResult:${tabId}`; }
async function saveTabResult(tabId, state) {
  await chrome.storage.local.set({[resultKey(tabId)]: state});
}
async function getTabResult(tabId) {
  const stored = await chrome.storage.local.get(resultKey(tabId));
  return stored[resultKey(tabId)] || null;
}

async function post(path, payload) {
  const response = await fetch(`${API}${path}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload), signal: AbortSignal.timeout(8000)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `API returned ${response.status}`);
  return data;
}
function checkUrl(url) { return post("/api/check-url", {url}); }
function checkEmail(subject, body) { return post("/api/check-email", {subject, body}); }
