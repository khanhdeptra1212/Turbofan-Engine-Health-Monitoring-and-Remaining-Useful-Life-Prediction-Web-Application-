const chatWidget = document.getElementById('ai-chat-widget');
const launcher = document.getElementById('ai-assistant-launcher');
const bubble = document.getElementById('ai-assistant-bubble');
const chatMessages = document.getElementById('ai-chat-messages');
const chatInput = document.getElementById('ai-chat-input');

let sessionId = sessionStorage.getItem('kdz_chat_session_id');
if (!sessionId) {
  sessionId = 'session_' + Math.random().toString(36).slice(2);
  sessionStorage.setItem('kdz_chat_session_id', sessionId);
}

let bubbleTimer = null;
let shakeTimer = null;

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLauncherBubbleTemporarily() {
  bubble.classList.remove('hidden');
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => {
    if (chatWidget.classList.contains('collapsed')) {
      bubble.classList.add('hidden');
    }
  }, 2800);
}

function triggerMascotShake() {
  launcher.classList.remove('active');
  void launcher.offsetWidth;
  launcher.classList.add('active');
  showLauncherBubbleTemporarily();
}

function startMascotAutoHint() {
  clearInterval(shakeTimer);
  shakeTimer = setInterval(() => {
    if (chatWidget.classList.contains('collapsed')) {
      triggerMascotShake();
    }
  }, 12000);
}

function toggleChatWidget() {
  const isCollapsed = chatWidget.classList.contains('collapsed');
  if (isCollapsed) {
    chatWidget.classList.remove('collapsed');
    chatWidget.classList.add('expanded');
    bubble.classList.add('hidden');
    setTimeout(() => chatInput.focus(), 180);
  } else {
    minimizeChatWidget();
  }
}

function minimizeChatWidget() {
  chatWidget.classList.remove('expanded');
  chatWidget.classList.add('collapsed');
  showLauncherBubbleTemporarily();
}

async function sendChatMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage('user', message);
  chatInput.value = '';
  addMessage('bot', 'Đang suy nghĩ...');

  const pageType = document.body.dataset.page || 'unknown';
  const engineId = document.body.dataset.engineId || '';

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        session_id: sessionId,
        current_page: pageType,
        current_engine_id: engineId
      })
    });

    const data = await response.json();
    chatMessages.lastChild.remove();
    addMessage('bot', data.answer || 'Không có phản hồi.');
  } catch (error) {
    chatMessages.lastChild.remove();
    addMessage('bot', 'Lỗi khi gọi chat backend.');
  }
}

function quickAsk(text) {
  if (chatWidget.classList.contains('collapsed')) {
    toggleChatWidget();
  }
  chatInput.value = text;
  sendChatMessage();
}

chatInput.addEventListener('keydown', function (e) {
  if (e.key === 'Enter') {
    sendChatMessage();
  }
});

addMessage('bot', 'Xin chào, tôi là KDZ Assistant. Bạn có thể hỏi về engine, RUL, warning hoặc critical.');
showLauncherBubbleTemporarily();
startMascotAutoHint();