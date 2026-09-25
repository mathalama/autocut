/**
 * AutoCut Live Subtitles Client for OBS Browser Source
 * Production-ready with 2-line auto-scroll and auto-reconnect
 */

(function () {
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const captionContainer = document.getElementById("caption-container");
  const captionCard = document.getElementById("caption-card");
  const captionPrev = document.getElementById("caption-prev");
  const captionText = document.getElementById("caption-text") || document.getElementById("caption-committed");

  // Read URL query parameters for customization
  const urlParams = new URLSearchParams(window.location.search);
  const theme = urlParams.get("theme") || "standard";
  document.body.className = `theme-${theme}`;

  const customSize = urlParams.get("size");
  if (customSize) {
    document.documentElement.style.setProperty("--font-size", `${parseInt(customSize, 10)}px`);
  }

  const customColor = urlParams.get("color");
  if (customColor) {
    document.documentElement.style.setProperty("--text-color", customColor);
  }

  const customBg = urlParams.get("bg");
  if (customBg) {
    const resolvedBg = (customBg === "none" || customBg === "transparent") ? "transparent" : customBg;
    document.documentElement.style.setProperty("--bg-color", resolvedBg);
  }

  const customAlign = urlParams.get("align");
  if (customAlign) {
    captionCard.style.textAlign = customAlign;
  }

  // Position control: ?pos=top | ?pos=center | ?pos=bottom (default)
  const posParam = urlParams.get("pos");
  if (posParam === "top") {
    captionContainer.style.bottom = "auto";
    captionContainer.style.top = "40px";
  } else if (posParam === "center") {
    captionContainer.style.bottom = "auto";
    captionContainer.style.top = "50%";
    captionContainer.style.transform = "translateY(-50%)";
  }

  if (urlParams.get("hideStatus") === "1" || urlParams.get("stream") === "1") {
    statusPill.classList.add("hide");
  }

  // Max words to display simultaneously (keeps subtitles clean within 1-2 lines)
  const MAX_DISPLAY_WORDS = 14;

  const wsHost = window.location.host || "127.0.0.1:8765";
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${wsHost}/ws`;

  let ws = null;
  let reconnectTimeout = null;
  let fadeTimeout = null;

  function setStatus(state, msg) {
    if (!statusPill) return;
    statusPill.className = `status-pill ${state}`;
    if (urlParams.get("status") !== "1") {
      statusPill.classList.add("hide");
    }
    if (statusText) {
      statusText.textContent = msg;
    }
  }

  function clearCaptions() {
    captionCard.classList.add("fading");
    setTimeout(() => {
      if (captionCard.classList.contains("fading")) {
        if (captionPrev) captionPrev.textContent = "";
        if (captionText) captionText.textContent = "";
        captionCard.classList.add("empty");
        captionCard.classList.remove("fading");
      }
    }, 400);
  }

  function trimToMaxWords(text) {
    const words = text.trim().split(/\s+/);
    if (words.length > MAX_DISPLAY_WORDS) {
      return "…" + words.slice(words.length - MAX_DISPLAY_WORDS).join(" ");
    }
    return text;
  }

  function updateCaptions(data) {
    if (fadeTimeout) {
      clearTimeout(fadeTimeout);
      fadeTimeout = null;
    }

    if (data.type === "clear") {
      clearCaptions();
      return;
    }

    if (data.type === "status") {
      if (data.paused) {
        setStatus("error", "PAUSED");
        clearCaptions();
      } else {
        setStatus("connected", "Live Subtitles Ready");
      }
      return;
    }

    captionCard.classList.remove("empty", "fading");
    const clean = trimToMaxWords(data.text);
    if (!clean) return;

    // Shift previous sentence to upper line, and present new finalized text
    if (captionPrev && captionText && captionText.textContent && captionText.textContent !== clean) {
      captionPrev.textContent = captionText.textContent;
    }
    if (captionText) {
      captionText.textContent = clean;
    }
  }

  function connect() {
    setStatus("connecting", "Connecting...");
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      setStatus("connected", "Live Subtitles Ready");
      setTimeout(() => {
        statusPill.classList.add("hide");
      }, 3500);
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        updateCaptions(data);
      } catch (err) {
        console.error("Failed to parse caption message:", err);
      }
    };

    ws.onclose = function () {
      setStatus("error", "Disconnected");
      statusPill.classList.remove("hide");
      scheduleReconnect();
    };

    ws.onerror = function () {
      setStatus("error", "Connection Error");
      ws.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimeout) return;
    reconnectTimeout = setTimeout(() => {
      reconnectTimeout = null;
      connect();
    }, 2000);
  }

  connect();
})();
