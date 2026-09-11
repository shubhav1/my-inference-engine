from html import escape
from urllib.parse import quote

from flask import Flask, Response, redirect, request, stream_with_context

from chat import Chat, MODEL_PATH, load_qwen
from tokenizer import QwenTokenizer

app = Flask(__name__)

print("loading tokenizer and model...")
tokenizer = QwenTokenizer(MODEL_PATH)
model = load_qwen(MODEL_PATH)
chats = {}

STYLE = """
* { box-sizing: border-box; }
body {
  margin: 0;
  display: flex;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1e1e2e;
  background: #ffffff;
}
nav.sidebar {
  width: 240px;
  flex-shrink: 0;
  background: #1e1e2e;
  color: #cdd6f4;
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
}
nav.sidebar h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #7f849c;
  margin: 4px 8px 12px;
}
nav.sidebar a {
  display: block;
  padding: 10px 12px;
  border-radius: 8px;
  color: #cdd6f4;
  text-decoration: none;
  font-size: 14px;
  margin-bottom: 2px;
}
nav.sidebar a:hover { background: #313244; }
nav.sidebar a.active { background: #45475a; color: #fff; font-weight: 600; }
nav.sidebar .empty { color: #7f849c; font-size: 13px; padding: 10px 12px; }
nav.sidebar form {
  margin-top: auto;
  display: flex;
  gap: 6px;
}
nav.sidebar input[type=text] {
  flex: 1;
  min-width: 0;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid #45475a;
  background: #313244;
  color: #cdd6f4;
  font-size: 13px;
}
nav.sidebar button {
  border: none;
  background: #89b4fa;
  color: #1e1e2e;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
header.topbar {
  padding: 16px 24px;
  border-bottom: 1px solid #ececf1;
  font-weight: 600;
  font-size: 15px;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.placeholder {
  margin: auto;
  color: #9a9aa8;
  font-size: 14px;
  text-align: center;
}
.row { display: flex; }
.row.user { justify-content: flex-end; }
.row.assistant { justify-content: flex-start; }
.bubble {
  max-width: 65%;
  padding: 10px 14px;
  border-radius: 16px;
  font-size: 14.5px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.row.user .bubble { background: #89b4fa; color: #1e1e2e; border-bottom-right-radius: 4px; }
.row.assistant .bubble { background: #f0f0f4; color: #1e1e2e; border-bottom-left-radius: 4px; }
form.composer {
  display: flex;
  gap: 10px;
  padding: 16px 24px;
  border-top: 1px solid #ececf1;
}
form.composer input[type=text] {
  flex: 1;
  padding: 12px 16px;
  border-radius: 22px;
  border: 1px solid #ddd;
  font-size: 14.5px;
}
form.composer input[type=text]:focus { outline: 2px solid #89b4fa; }
form.composer button {
  border: none;
  background: #1e1e2e;
  color: #fff;
  border-radius: 22px;
  padding: 0 22px;
  font-size: 14.5px;
  font-weight: 600;
  cursor: pointer;
}
.hint { padding: 0 24px 12px; margin: -8px 0 0; color: #9a9aa8; font-size: 12px; }
"""


def sidebar(active=None):
    if chats:
        links = "".join(
            f'<a class="{"active" if name == active else ""}" href="/chat/{quote(name)}">{escape(name)}</a>'
            for name in chats
        )
    else:
        links = '<div class="empty">no chats yet</div>'
    return f"""
<nav class="sidebar">
  <h2>Chats</h2>
  {links}
  <form method="post" action="/new">
    <input type="text" name="name" placeholder="new chat" required>
    <button type="submit">+</button>
  </form>
</nav>
"""


def page(active, main_html):
    return f"""<!doctype html>
<title>chat</title>
<style>{STYLE}</style>
{sidebar(active)}
<main>{main_html}</main>"""


def bubble(role, text):
    return f'<div class="row {role}"><div class="bubble">{escape(text)}</div></div>'


def chat_head(name):
    return f"""<!doctype html>
<title>chat</title>
<style>{STYLE}</style>
{sidebar(name)}
<main>
<header class="topbar">{escape(name)}</header>
<div class="messages">"""


def chat_tail(name):
    return f"""</div>
<p class="hint">no kv-cache &mdash; replies stream in as they're generated, but can take a while</p>
<form class="composer" method="post" action="/chat/{quote(name)}/send">
  <input type="text" name="message" placeholder="Message..." required autofocus>
  <button type="submit">Send</button>
</form>
</main>"""


@app.get("/")
def index():
    return page(None, """
<header class="topbar">Select a chat</header>
<div class="messages"><div class="placeholder">Pick a chat on the left, or start a new one.</div></div>
""")


@app.post("/new")
def new_chat():
    name = request.form["name"].strip()
    if name and name not in chats:
        chats[name] = Chat(tokenizer)
    return redirect(f"/chat/{quote(name)}")


@app.get("/chat/<name>")
def view_chat(name):
    if name not in chats:
        return redirect("/")
    bubbles = "".join(bubble(role, text) for role, text in chats[name].messages)
    bubbles = bubbles or '<div class="placeholder">Say something to start the conversation.</div>'
    return chat_head(name) + bubbles + chat_tail(name)


@app.post("/chat/<name>/send")
def send(name):
    if name not in chats:
        return redirect("/")
    message = request.form["message"]
    chat = chats[name]
    prior = "".join(bubble(role, text) for role, text in chat.messages)

    def stream():
        yield chat_head(name)
        yield prior
        yield bubble("user", message)
        yield '<div class="row assistant"><div class="bubble">'
        for piece in chat.send_stream(model, message):
            yield escape(piece)
        yield "</div></div>"
        yield chat_tail(name)

    return Response(stream_with_context(stream()), mimetype="text/html")


if __name__ == "__main__":
    app.run(debug=False)
