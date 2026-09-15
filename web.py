from html import escape
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from chat import Chat, MODEL_PATH, load_qwen
from tokenizer import QwenTokenizer

app = FastAPI()

print("loading tokenizer and model...")
tokenizer = QwenTokenizer(MODEL_PATH)
model = load_qwen(MODEL_PATH)
chats = {}

STYLE = """
:root {
  --bg: #faf9f6;
  --bg-panel: #ffffff;
  --bg-sidebar: #f1eee7;
  --border: #e6e1d6;
  --text: #2b2a27;
  --text-muted: #8c8677;
  --accent: #6c63ff;
  --accent-hover: #5a52e6;
  --accent-soft: #ecebff;
  --danger: #d6555b;
  --danger-soft: #fbeaea;
  --danger-soft-hover: #f6d9da;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  display: flex;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}
nav.sidebar {
  width: 240px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  padding: 20px 14px;
  display: flex;
  flex-direction: column;
}
nav.sidebar h2 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  font-weight: 700;
  margin: 4px 8px 14px;
}
nav.sidebar a {
  display: block;
  padding: 10px 12px;
  border-radius: 10px;
  color: var(--text);
  text-decoration: none;
  font-size: 14px;
  margin-bottom: 4px;
  transition: background 0.15s ease;
}
nav.sidebar a:hover { background: var(--accent-soft); }
nav.sidebar a.active { background: var(--accent); color: #fff; font-weight: 600; }
nav.sidebar .empty { color: var(--text-muted); font-size: 13px; padding: 10px 12px; }
nav.sidebar form {
  margin-top: auto;
  padding-top: 12px;
  display: flex;
  gap: 6px;
}
nav.sidebar input[type=text] {
  flex: 1;
  min-width: 0;
  padding: 9px 10px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg-panel);
  color: var(--text);
  font-size: 13px;
  font-family: inherit;
}
nav.sidebar input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.btn {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 8px 14px;
  font-size: 13px;
  font-weight: 600;
  font-family: inherit;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent-hover); }
.btn-secondary { background: var(--bg-panel); color: var(--text); border-color: var(--border); }
.btn-secondary:hover { background: var(--bg-sidebar); }
.btn-danger { background: var(--danger-soft); color: var(--danger); }
.btn-danger:hover { background: var(--danger-soft-hover); }
main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg-panel);
}
header.topbar {
  padding: 18px 28px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  font-size: 15px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.topbar-actions { display: flex; gap: 8px; }
.topbar-actions form { margin: 0; }
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.placeholder {
  margin: auto;
  color: var(--text-muted);
  font-size: 14px;
  text-align: center;
}
.row { display: flex; }
.row.user { justify-content: flex-end; }
.row.assistant { justify-content: flex-start; }
.bubble {
  max-width: 65%;
  padding: 11px 16px;
  border-radius: 18px;
  font-size: 14.5px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.row.user .bubble { background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
.row.assistant .bubble { background: var(--bg-sidebar); color: var(--text); border-bottom-left-radius: 4px; }
form.composer {
  display: flex;
  gap: 10px;
  padding: 18px 28px;
  border-top: 1px solid var(--border);
}
form.composer input[type=text] {
  flex: 1;
  padding: 12px 18px;
  border-radius: 24px;
  border: 1px solid var(--border);
  font-size: 14.5px;
  font-family: inherit;
  background: var(--bg);
}
form.composer input[type=text]:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
form.composer button {
  border: none;
  background: var(--accent);
  color: #fff;
  border-radius: 24px;
  padding: 0 24px;
  font-size: 14.5px;
  font-weight: 600;
  font-family: inherit;
  cursor: pointer;
  transition: background 0.15s ease;
}
form.composer button:hover { background: var(--accent-hover); }
.hint { padding: 0 28px 14px; margin: -10px 0 0; color: var(--text-muted); font-size: 12px; }
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
    <button type="submit" class="btn btn-primary">+</button>
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
<header class="topbar">
  <span>{escape(name)}</span>
  <div class="topbar-actions">
    <form method="post" action="/chat/{quote(name)}/clear">
      <button type="submit" class="btn btn-secondary">Clear</button>
    </form>
    <form method="post" action="/chat/{quote(name)}/delete">
      <button type="submit" class="btn btn-danger">Delete</button>
    </form>
  </div>
</header>
<div class="messages">
<script>
(function() {{
  var el = document.currentScript.parentElement;
  var scroll = function() {{ el.scrollTop = el.scrollHeight; }};
  scroll();
  new MutationObserver(scroll).observe(el, {{childList: true, subtree: true, characterData: true}});
}})();
</script>"""


def chat_tail(name):
    return f"""</div>
<form class="composer" method="post" action="/chat/{quote(name)}/send">
  <input type="text" name="message" placeholder="Message..." required autofocus>
  <button type="submit">Send</button>
</form>
</main>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return page(None, """
<header class="topbar">Select a chat</header>
<div class="messages"><div class="placeholder">Pick a chat on the left, or start a new one.</div></div>
""")


@app.post("/new")
def new_chat(name: str = Form(...)):
    name = name.strip()
    if name and name not in chats:
        chats[name] = Chat(tokenizer)
    return RedirectResponse(f"/chat/{quote(name)}", status_code=303)


@app.get("/chat/{name}", response_class=HTMLResponse)
def view_chat(name: str):
    if name not in chats:
        return RedirectResponse("/", status_code=303)
    chat = chats[name]

    catch_up, live_queue = chat.watch()
    if live_queue is None:
        bubbles = "".join(bubble(role, text) for role, text in chat.all_messages())
        bubbles = bubbles or '<div class="placeholder">Say something to start the conversation.</div>'
        return chat_head(name) + bubbles + chat_tail(name)

    # a reply is actively streaming right now -- join it live instead of a static snapshot
    prior = "".join(bubble(role, text) for role, text in chat.messages)

    def stream():
        yield chat_head(name)
        yield prior
        yield '<div class="row assistant"><div class="bubble">'
        yield escape(catch_up)
        while True:
            piece = live_queue.get()
            if piece is None:
                break
            yield escape(piece)
        yield "</div></div>"
        yield "".join(bubble("user", text) for text in chat.pending)  # any messages queued behind it
        yield chat_tail(name)

    return StreamingResponse(stream(), media_type="text/html")


@app.post("/chat/{name}/clear")
def clear_chat(name: str):
    if name in chats:
        chats[name] = Chat(tokenizer)
    return RedirectResponse(f"/chat/{quote(name)}", status_code=303)


@app.post("/chat/{name}/delete")
def delete_chat(name: str):
    chats.pop(name, None)
    return RedirectResponse("/", status_code=303)


@app.post("/chat/{name}/send")
def send(name: str, message: str = Form(...)):
    if name not in chats:
        return RedirectResponse("/", status_code=303)
    chat = chats[name]
    prior = "".join(bubble(role, text) for role, text in chat.all_messages())

    def stream():
        yield chat_head(name)
        yield prior
        yield bubble("user", message)
        yield '<div class="row assistant"><div class="bubble">'
        for piece in chat.send_stream(model, message):
            yield escape(piece)
        yield "</div></div>"
        yield chat_tail(name)

    return StreamingResponse(stream(), media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5050)  # 5000 collides with macOS AirPlay Receiver
