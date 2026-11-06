(() => {
  const selectors = window.AI_SECURITY_SELECTORS || {};
  let lastEmailKey = "";
  let emailTimer = 0;

  function firstVisible(selectorList) {
    for (const selector of selectorList || []) {
      const nodes = [...document.querySelectorAll(selector)];
      const node = nodes.reverse().find(item => item.innerText?.trim() && item.getClientRects().length > 0);
      if (node) return node;
    }
    return null;
  }
  function getVisibleEmail() {
    const subjectNode = firstVisible(selectors.subject);
    const bodyNode = firstVisible(selectors.messageBody);
    const subject = subjectNode?.innerText?.trim() || "";
    const body = bodyNode?.innerText?.trim() || "";
    if (!body || body.length < 8) return null;
    const message = body.slice(0, 200000);
    return {subject: subject.slice(0, 500), body: message};
  }
  function addBanner(id, heading, result, severity, inline = false) {
    document.getElementById(id)?.remove();
    const box = document.createElement("aside");
    box.id = id;
    box.setAttribute("role", "alert");
    const title = document.createElement("strong"); title.textContent = heading;
    const score = document.createElement("span"); score.textContent = `Risk score: ${result.risk_score}/100 · ${result.prediction.toUpperCase()}`;
    const reason = document.createElement("span"); reason.textContent = (result.reasons || []).slice(0, 2).join(" · ");
    const close = document.createElement("button"); close.textContent = "Dismiss"; close.setAttribute("aria-label", "Dismiss security warning");
    close.onclick = () => box.remove();
    box.append(title, score, reason, close);
    Object.assign(box.style, inline
      ? {display:"flex",alignItems:"center",gap:"10px",margin:"8px 0",padding:"10px 12px",borderRadius:"8px",background:severity==="malicious"?"#8b1010":"#8a5a00",color:"#fff",font:"13px Arial,sans-serif"}
      : {position:"fixed",top:"0",left:"0",right:"",zIndex:"2147483647",display:"flex",alignItems:"center",gap:"14px",padding:"12px 16px",background:severity==="malicious"?"#8b1010":"#8a5a00",color:"#fff",font:"14px Arial,sans-serif",boxShadow:"0 3px 16px #0005"});
    close.style.cssText = "margin-left:auto;padding:6px 10px;border:1px solid #fff8;border-radius:6px;background:transparent;color:white;cursor:pointer";
    if (inline) {
      const message = firstVisible(selectors.messageContainer);
      if (message?.parentElement) message.parentElement.insertBefore(box, message.nextSibling);
      else document.documentElement.appendChild(box);
    } else document.documentElement.appendChild(box);
  }
  function scanOpenedEmail() {
    if (location.hostname !== selectors.gmailHost) return;
    const email = getVisibleEmail();
    if (!email) return;
    const key = `${email.subject}\u0000${email.body}`;
    if (key === lastEmailKey) return;
    lastEmailKey = key;
    chrome.runtime.sendMessage({type:"AUTO_CHECK_EMAIL", ...email}).catch(() => {});
  }
  if (location.hostname === selectors.gmailHost) {
    const observer = new MutationObserver(() => {
      clearTimeout(emailTimer);
      emailTimer = setTimeout(scanOpenedEmail, 900);
    });
    observer.observe(document.body, {subtree:true, childList:true, characterData:true});
    setTimeout(scanOpenedEmail, 1200);
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "URL_WARNING") {
      addBanner("__ai_security_url_warning", "⚠ Potential phishing website", message.result, message.result.prediction);
    } else if (message?.type === "EMAIL_WARNING") {
      addBanner("__ai_security_email_warning", "⚠ Potential phishing email", message.result, message.result.prediction, true);
    } else if (message?.type === "GET_VISIBLE_EMAIL") {
      sendResponse(getVisibleEmail() || {});
    }
  });
})();
