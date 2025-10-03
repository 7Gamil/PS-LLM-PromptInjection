# flask_xss_demo.py
# Simple Flask app that serves a Bootstrap page and demonstrates a chat-style
# interaction with an LLM.
# WARNING: This intentionally contains XSS vulnerabilities for demo/teaching only.
# The LLM output is reflected into the page without escaping. Do NOT deploy.
from chat.EmailTemplates import EmailTemplates
from flask import Flask, request, Response, render_template_string

app = Flask(__name__)

PAGE = '''<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Chat Demo (LLM)</title>
    <!-- Bootstrap CSS CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      .chat { max-height: 60vh; overflow:auto; }
      .entry { margin-bottom: .75rem; }
      .user { text-align: right; }
      .assistant { text-align: left; }
      .bubble { display:inline-block; padding:.5rem .75rem; border-radius:12px; }
      .user .bubble { background:#0d6efd; color:white; }
      .assistant .bubble { background:#e9ecef; color:#212529; }
    </style>
  </head>
  <body class="bg-light">
    <div class="container py-4">
      <div class="row justify-content-center">
        <div class="col-md-8">
          <h3 class="mb-3">LLM Chat</h3>

          <div class="card mb-3">
            <div class="card-body chat" id="chat">
              <!-- Chat entries will be appended here -->
            </div>
          </div>

          <form id="chatForm" class="d-flex gap-2">
            <input id="msg" class="form-control" placeholder='Type a message '>
            <button class="btn btn-primary">Send</button>
          </form>

          <div class="mt-3 text-muted small">This demo creates email templates. </div>
        </div>
      </div>
    </div>

    <script>
      const chat = document.getElementById('chat');
      const form = document.getElementById('chatForm');
      const msgInput = document.getElementById('msg');

      function appendText(text, className) {
        const entry = document.createElement('div');
        entry.className = className;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = htmlEncode(text); 

        entry.appendChild(bubble);
        chat.appendChild(entry);

        chat.scrollTop = chat.scrollHeight;
      }

      function htmlEncode(str) {
          return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
      }
      
      function appendAssistant(text) {
        appendText(text, 'entry assistant')
      }

      function appendUser(text) {
        appendText(text, 'entry user')
      }

      async function sendMessage(text) {
        appendUser(text);

        // Send to server;
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ message: text })
        });

        const reply = await resp.text();

        appendAssistant(reply);
      }

      form.addEventListener('submit', function (e) {
        e.preventDefault();
        const text = msgInput.value || '';
        if (!text) return;
        msgInput.value = '';
        sendMessage(text);
      });

    </script>

    <!-- Bootstrap JS bundle (optional) -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
'''


def generate_llm_response(user_text: str) -> str:
    chat = EmailTemplates()
    chat.run_example(user_text)
    return chat.messages[-1]["content"]


@app.route('/', methods=['GET'])
def index():
    return render_template_string(PAGE)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    user_msg = request.form.get('message', '')
    reply = generate_llm_response(user_msg)
    return Response(reply, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
