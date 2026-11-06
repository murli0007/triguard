// Gmail selectors are deliberately isolated because Gmail's DOM may change.
window.AI_SECURITY_SELECTORS = {
  gmailHost: "mail.google.com",
  subject: ["h2.hP", "[role='main'] h2", "[data-legacy-thread-id] h2"],
  messageBody: [".a3s.aiL", ".a3s", "[role='list'] [dir='ltr']"],
  messageContainer: [".adn.ads", "[role='main'] .h7", "[role='main'] [data-message-id]"]
};
