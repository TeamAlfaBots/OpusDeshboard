"""
Owner-only chat monitor (web page) served on the same port as /health.

  GET  /admin                 login form, or the dashboard once logged in
  POST /admin/login           password check (rate limited), sets a signed HttpOnly cookie
  POST /admin/logout
  GET  /admin/api/users       users per chat: last activity, message count, flags
  GET  /admin/api/messages    messages (filter: chat_id, user_id, q, flag, before, limit)

Off unless ADMIN_PASSWORD is set. Data comes from the masked `chat_log` collection (auto-deleted after LOG_DAYS).
All chat text is shown with textContent (never innerHTML), so a malicious message cannot run script in your browser.
"""
import hashlib
import hmac
import json
import secrets
import time
from collections import deque
from datetime import datetime, timezone

from config import (
    ADMIN_LOGIN_MAX_ATTEMPTS,
    ADMIN_LOGIN_WINDOW,
    ADMIN_PASSWORD,
    ADMIN_SESSION_HOURS,
    API_HASH,
    DASHBOARD_MAX_PAGE,
    DASHBOARD_PAGE_SIZE,
)
from memory import FLAGS
from util import BoundedDict

COOKIE = "monitor_session"
BASE = "/admin"


# ------------------------------------------------------------------ page templates
_STYLE = """
:root{color-scheme:dark light;--bg:#0f1115;--card:#171a21;--line:#262b36;--text:#e8eaf0;--mute:#8b93a7;--me:#2b6cff;--bot:#262b36;--accent:#6ea8ff}
@media (prefers-color-scheme:light){:root{--bg:#f4f5f8;--card:#fff;--line:#e2e5ec;--text:#161922;--mute:#667085;--bot:#e8ebf2;--accent:#2b6cff}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:8px;padding:10px 12px;background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0;flex:1}button,input{font:inherit;color:inherit}
button{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:7px 12px;cursor:pointer}
button.active{background:var(--accent);border-color:var(--accent);color:#fff}
.bar{padding:10px 12px;display:flex;flex-direction:column;gap:8px}.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{border-radius:999px;padding:5px 11px;font-size:13px}
input[type=search],input[type=password]{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card)}
main{padding:0 12px 40px;max-width:760px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:11px 13px;margin:8px 0;cursor:pointer}
.row{display:flex;justify-content:space-between;gap:8px;align-items:baseline}.name{font-weight:600}.mute{color:var(--mute);font-size:13px}
.badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.f-love{background:#ff4d8d22;border-color:#ff4d8d}.f-real{background:#ffb02022;border-color:#ffb020}.f-meet{background:#a855f722;border-color:#a855f7}
.f-minor{background:#ff3b3b33;border-color:#ff3b3b}.f-distress{background:#ff3b3b33;border-color:#ff3b3b}.f-money{background:#22c55e22;border-color:#22c55e}
.msg{max-width:84%;padding:8px 11px;border-radius:14px;margin:5px 0;white-space:pre-wrap;word-break:break-word}
.in{background:var(--bot);border-bottom-left-radius:4px}.out{background:var(--me);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
.meta{font-size:11px;color:var(--mute);margin:8px 2px 0}.out+.meta{text-align:right}.who{font-weight:600;margin-top:12px}
.empty,.err{padding:24px;text-align:center;color:var(--mute)}.err{color:#ff6b6b}
.login{max-width:340px;margin:18vh auto;padding:0 16px}.login form{display:flex;flex-direction:column;gap:10px}
.login .err{padding:0;text-align:left}
"""

_LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>Sign in</title><style nonce="__NONCE__">__STYLE__</style></head><body>
<div class="login"><h1>Chat monitor</h1><p class="mute">Owner only.</p>
<form method="post" action="/admin/login"><input type="password" name="password" placeholder="Password" autocomplete="current-password" autofocus required>
<button class="active" type="submit">Sign in</button>__ERROR__</form></div></body></html>"""

_APP_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>Chat monitor</title><style nonce="__NONCE__">__STYLE__</style></head><body>
<header><h1 id="title">Chat monitor</h1><button id="back" hidden>← Back</button><button id="auto" title="Auto refresh">⟳ Auto</button><button id="logout">Logout</button></header>
<div class="bar" id="bar">
  <div class="chips" id="tabs"><button class="chip active" data-tab="users">Users</button><button class="chip" data-tab="feed">Live feed</button></div>
  <div class="chips" id="flags"></div>
  <input type="search" id="q" placeholder="Search name or message…" autocomplete="off">
</div>
<main id="list"></main>
<script nonce="__NONCE__">
const FLAGS = __FLAGS__;
const $ = (s) => document.querySelector(s);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
const S = { tab: "users", flag: "", q: "", conv: null, before: null, timer: null, auto: false, seq: 0 };

async function api(path) {
  const r = await fetch(path, { credentials: "same-origin", headers: { "Accept": "application/json" } });
  if (r.status === 401) { location.href = "/admin"; throw new Error("auth"); }
  if (!r.ok) throw new Error("http " + r.status);
  return r.json();
}
function fmt(ts) { try { return new Date(ts).toLocaleString([], { dateStyle: "short", timeStyle: "short" }); } catch (e) { return String(ts); } }
function badges(flags) {
  const b = el("div", "badges");
  (flags || []).forEach((f) => b.appendChild(el("span", "badge f-" + f.replace(/[^a-z]/g, ""), f)));
  return b;
}
function note(list, text, cls) { list.replaceChildren(el("div", cls || "empty", text)); }

function renderFlagChips() {
  const box = $("#flags"); box.replaceChildren();
  [["", "All"]].concat(FLAGS.map((f) => [f, f])).forEach(([v, label]) => {
    const c = el("button", "chip" + (S.flag === v ? " active" : ""), label);
    c.onclick = () => { S.flag = v; renderFlagChips(); refresh(); };
    box.appendChild(c);
  });
}
async function loadUsers() {
  const list = $("#list"); const my = ++S.seq;
  const d = await api("/admin/api/users?q=" + encodeURIComponent(S.q) + "&flag=" + encodeURIComponent(S.flag));
  if (my !== S.seq) return;
  if (!d.users.length) return note(list, "No chats yet.");
  list.replaceChildren();
  d.users.forEach((u) => {
    const card = el("div", "card");
    const row = el("div", "row");
    row.appendChild(el("span", "name", (u.name || "Unknown") + (u.chat_type === "group" && u.chat_title ? "  ·  " + u.chat_title : "")));
    row.appendChild(el("span", "mute", fmt(u.last_ts)));
    card.appendChild(row);
    card.appendChild(el("div", "mute", (u.username ? "@" + u.username + "  ·  " : "") + u.count + " msgs  ·  " + u.last_text));
    if (u.flags && u.flags.length) card.appendChild(badges(u.flags));
    card.onclick = () => openConv(u);
    list.appendChild(card);
  });
}
function addMessages(list, msgs, withWho) {
  msgs.forEach((m) => {
    if (withWho) list.appendChild(el("div", "who", (m.name || "Unknown") + (m.chat_type === "group" && m.chat_title ? "  ·  " + m.chat_title : "")));
    list.appendChild(el("div", "msg in", m.user_text));
    list.appendChild(el("div", "meta", fmt(m.ts)));
    if (m.bot_text) list.appendChild(el("div", "msg out", m.bot_text));
    else list.appendChild(el("div", "meta", m.status === "ai_down" ? "(AI was down - no reply)" : "(no reply sent)"));
    if (m.flags && m.flags.length) list.appendChild(badges(m.flags));
  });
}
async function loadFeed() {
  const list = $("#list"); const my = ++S.seq;
  const d = await api("/admin/api/messages?q=" + encodeURIComponent(S.q) + "&flag=" + encodeURIComponent(S.flag));
  if (my !== S.seq) return;
  if (!d.messages.length) return note(list, "Nothing here.");
  list.replaceChildren(); addMessages(list, d.messages, true);
}
async function openConv(u) {
  S.conv = u; S.before = null;
  $("#bar").hidden = true; $("#back").hidden = false; $("#title").textContent = (u.name || "Chat");
  const list = $("#list"); list.replaceChildren();
  await moreConv(true);
}
async function moreConv(first) {
  const list = $("#list");
  const u = S.conv; if (!u) return;
  const path = "/admin/api/messages?chat_id=" + u.chat_id + "&user_id=" + u.user_id + (S.before ? "&before=" + encodeURIComponent(S.before) : "");
  const d = await api(path);
  const msgs = d.messages.slice().reverse();               // oldest first inside one page
  const old = list.querySelector(".more"); if (old) old.remove();
  const frag = document.createElement("div"); addMessages(frag, msgs, false);
  if (first) list.replaceChildren(frag); else list.insertBefore(frag, list.firstChild);
  if (d.next_before) {
    S.before = d.next_before;
    const more = el("button", "more", "Load older"); more.onclick = () => moreConv(false);
    list.insertBefore(more, list.firstChild);
  }
  if (first) window.scrollTo(0, document.body.scrollHeight);
}
function back() { S.conv = null; $("#bar").hidden = false; $("#back").hidden = true; $("#title").textContent = "Chat monitor"; refresh(); }
function refresh() {
  if (S.conv) return;
  (S.tab === "users" ? loadUsers() : loadFeed()).catch((e) => { if (e.message !== "auth") note($("#list"), "Could not load data. Retrying…", "err"); });
}
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => {
  S.tab = b.dataset.tab; document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b)); refresh();
});
let qt = null;
$("#q").addEventListener("input", (e) => { clearTimeout(qt); qt = setTimeout(() => { S.q = e.target.value.trim(); refresh(); }, 300); });
$("#back").onclick = back;
$("#auto").onclick = () => {
  S.auto = !S.auto; $("#auto").classList.toggle("active", S.auto);
  clearInterval(S.timer); if (S.auto) S.timer = setInterval(refresh, 15000);
};
$("#logout").onclick = async () => { try { await fetch("/admin/logout", { method: "POST", credentials: "same-origin" }); } finally { location.href = "/admin"; } };
renderFlagChips(); refresh();
</script></body></html>"""


def _nonce() -> str:
    return secrets.token_urlsafe(16)


# ------------------------------------------------------------------ dashboard
class Dashboard:
    def __init__(self, get_memory, password: str = ADMIN_PASSWORD, secret_seed: str = API_HASH,
                 max_attempts: int = ADMIN_LOGIN_MAX_ATTEMPTS, window: int = ADMIN_LOGIN_WINDOW,
                 session_hours: int = ADMIN_SESSION_HOURS, page_size: int = DASHBOARD_PAGE_SIZE,
                 max_page: int = DASHBOARD_MAX_PAGE):
        self.get_memory = get_memory
        self.password = password
        # changing the password (or API_HASH) invalidates every existing session
        self._key = hashlib.sha256(f"{password}|{secret_seed}".encode()).digest()
        self.max_attempts, self.window = max_attempts, window
        self.session_seconds = session_hours * 3600
        self.page_size, self.max_page = page_size, max_page
        self._attempts = BoundedDict(1000)      # ip -> deque of failed-login times

    # ---- session cookie
    def _sign(self, expiry: int) -> str:
        return hmac.new(self._key, str(expiry).encode(), hashlib.sha256).hexdigest()

    def make_cookie(self, now: "float | None" = None) -> str:
        expiry = int((now if now is not None else time.time()) + self.session_seconds)
        return f"{expiry}.{self._sign(expiry)}"

    def valid_cookie(self, value: "str | None", now: "float | None" = None) -> bool:
        try:
            expiry_s, sig = (value or "").split(".", 1)
            expiry = int(expiry_s)
        except ValueError:
            return False
        if expiry < (now if now is not None else time.time()):
            return False
        return hmac.compare_digest(sig, self._sign(expiry))

    def authed(self, request) -> bool:
        return self.valid_cookie(request.cookies.get(COOKIE))

    # ---- login throttling
    @staticmethod
    def client_ip(request) -> str:
        forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return forwarded or getattr(request, "remote", None) or "?"

    def locked(self, ip: str, now: "float | None" = None) -> bool:
        now = time.time() if now is None else now
        q = self._attempts.get(ip)
        if not q:
            return False
        while q and q[0] < now - self.window:
            q.popleft()
        return len(q) >= self.max_attempts

    def record_failure(self, ip: str, now: "float | None" = None) -> None:
        q = self._attempts.get(ip)
        if q is None:
            q = self._attempts[ip] = deque()
        q.append(time.time() if now is None else now)

    # ---- responses
    @staticmethod
    def _headers(nonce: "str | None" = None) -> dict:
        csp = ("default-src 'none'; connect-src 'self'; form-action 'self'; base-uri 'none'; "
               "frame-ancestors 'none'; img-src 'self' data:")
        if nonce:
            csp += f"; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'"
        return {
            "Cache-Control": "no-store",
            "X-Robots-Tag": "noindex, nofollow",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": csp,
        }

    def _html(self, template: str, status: int = 200, error: str = ""):
        from aiohttp import web
        nonce = _nonce()
        body = (template.replace("__STYLE__", _STYLE).replace("__NONCE__", nonce)
                .replace("__FLAGS__", json.dumps(list(FLAGS)))
                .replace("__ERROR__", f'<div class="err">{error}</div>' if error else ""))
        return web.Response(text=body, status=status, content_type="text/html", headers=self._headers(nonce))

    def _json(self, data: dict, status: int = 200):
        from aiohttp import web
        return web.json_response(data, status=status, headers=self._headers())

    # ---- handlers
    async def page(self, request):
        return self._html(_APP_HTML if self.authed(request) else _LOGIN_HTML)

    async def login(self, request):
        from aiohttp import web
        ip = self.client_ip(request)
        if self.locked(ip):
            return self._html(_LOGIN_HTML, 429, "Too many attempts. Try again later.")
        try:
            form = await request.post()
            supplied = str(form.get("password", ""))
        except Exception:
            supplied = ""
        if not hmac.compare_digest(supplied.encode(), self.password.encode()):
            self.record_failure(ip)
            return self._html(_LOGIN_HTML, 401, "Wrong password.")
        self._attempts.pop(ip, None)
        resp = web.Response(status=303, headers={**self._headers(), "Location": BASE})
        secure = bool(getattr(request, "secure", False)) or request.headers.get("X-Forwarded-Proto") == "https"
        resp.set_cookie(COOKIE, self.make_cookie(), max_age=self.session_seconds, httponly=True,
                        secure=secure, samesite="Strict", path=BASE)
        return resp

    async def logout(self, request):
        from aiohttp import web
        resp = web.Response(status=303, headers={**self._headers(), "Location": BASE})
        resp.del_cookie(COOKIE, path=BASE)
        return resp

    def _parse_int(self, request, name: str):
        raw = request.query.get(name)
        if raw in (None, ""):
            return None, False
        try:
            return int(raw), False
        except ValueError:
            return None, True

    async def api_users(self, request):
        if not self.authed(request):
            return self._json({"error": "unauthorized"}, 401)
        mem = self.get_memory()
        if mem is None:
            return self._json({"error": "starting"}, 503)
        flag = request.query.get("flag", "")
        if flag and flag not in FLAGS:
            return self._json({"error": "bad flag"}, 400)
        users = await mem.dash_users(request.query.get("q", "").strip(), flag, self.page_size)
        if users is None:
            return self._json({"error": "database unavailable"}, 503)
        return self._json({"users": users})

    async def api_messages(self, request):
        if not self.authed(request):
            return self._json({"error": "unauthorized"}, 401)
        mem = self.get_memory()
        if mem is None:
            return self._json({"error": "starting"}, 503)
        chat_id, bad1 = self._parse_int(request, "chat_id")
        user_id, bad2 = self._parse_int(request, "user_id")
        limit, bad3 = self._parse_int(request, "limit")
        if bad1 or bad2 or bad3:
            return self._json({"error": "bad number"}, 400)
        limit = min(max(limit or self.page_size, 1), self.max_page)
        flag = request.query.get("flag", "")
        if flag and flag not in FLAGS:
            return self._json({"error": "bad flag"}, 400)
        before = None
        raw_before = request.query.get("before", "")
        if raw_before:
            try:
                before = datetime.fromisoformat(raw_before.replace("Z", "+00:00"))
                if before.tzinfo is None:
                    before = before.replace(tzinfo=timezone.utc)
            except ValueError:
                return self._json({"error": "bad date"}, 400)
        result = await mem.dash_messages(chat_id=chat_id, user_id=user_id, q=request.query.get("q", "").strip(),
                                         flag=flag, before=before, limit=limit)
        if result is None:
            return self._json({"error": "database unavailable"}, 503)
        messages, next_before = result
        return self._json({"messages": messages, "next_before": next_before})


def setup(app, get_memory, password: str = ADMIN_PASSWORD) -> "Dashboard | None":
    """Register the /admin routes. Does nothing (no routes at all) when no ADMIN_PASSWORD is set."""
    if not password:
        return None
    dash = Dashboard(get_memory, password=password)
    app.router.add_get(BASE, dash.page)
    app.router.add_post(BASE + "/login", dash.login)
    app.router.add_post(BASE + "/logout", dash.logout)
    app.router.add_get(BASE + "/api/users", dash.api_users)
    app.router.add_get(BASE + "/api/messages", dash.api_messages)
    return dash
