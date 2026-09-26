# WhatsApp AI Bot (Django)

A Django app that receives WhatsApp messages via the Meta Cloud API, hands them
off to a Celery worker, generates an AI reply, and sends it back.

## Project structure

```
whatsapp_ai_bot/
    manage.py
    requirements.txt
    .env.example
    myproject/          <- project config (settings, urls, celery app)
    whatsapp_bot/        <- the actual bot app (models, views, tasks, services)
    .vscode/              <- debug configs for VS Code
```

## 1. Open in VS Code

Open the `whatsapp_ai_bot` folder directly in VS Code (File → Open Folder).
Install the **Python** extension if you don't have it already.

## 2. Create a virtual environment

Open a terminal inside VS Code (`` Ctrl+` ``):

```bash
python -m venv venv
```

Activate it:
- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

Then select it as your interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `venv`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Set up environment variables

```bash
cp .env.example .env       # Mac/Linux
copy .env.example .env     # Windows
```

Open `.env` and fill in your real WhatsApp credentials (from Meta for Developers → your app → WhatsApp → API Setup).
`WHATSAPP_VERIFY_TOKEN` can be any random string you choose yourself — you'll enter the same value when registering the webhook on Meta's side.

## 5. Run migrations

```bash
python manage.py migrate
```

## 6. Install and start Redis (needed for Celery)

- **Mac:** `brew install redis && brew services start redis`
- **Windows:** use [Memurai](https://www.memurai.com/) or run Redis via Docker: `docker run -d -p 6379:6379 redis`
- **Linux:** `sudo apt install redis-server && sudo service redis-server start`

## 7. Run the app — two processes needed

**Option A — using VS Code's Run & Debug panel (recommended):**
Open the Run & Debug tab (`Ctrl+Shift+D`), pick **Django: runserver**, hit the green play button.
Then open a second Run & Debug session and pick **Celery: worker**.

**Option B — using terminals directly:**
```bash
# Terminal 1
python manage.py runserver

# Terminal 2 (new terminal: Ctrl+Shift+5)
celery -A myproject worker -l info --pool=solo
```
(Drop `--pool=solo` if you're on Mac/Linux — it's a Windows-only workaround.)

## 8. Expose your local server to the internet (for Meta's webhook)

Meta needs a public HTTPS URL to send messages to. Use [ngrok](https://ngrok.com/):

```bash
ngrok http 8000
```

Copy the `https://xxxx.ngrok-free.app` URL it gives you.

## 9. Register the webhook with Meta

In your Meta app dashboard → WhatsApp → Configuration:
- **Callback URL:** `https://xxxx.ngrok-free.app/api/whatsapp/webhook/`
- **Verify token:** same value as `WHATSAPP_VERIFY_TOKEN` in your `.env`
- Subscribe to the `messages` webhook field

## 10. Test it

Send a WhatsApp message to your test number from Meta's dashboard. You should see:
- The webhook log print in your `runserver` terminal
- The Celery worker pick up the task in its terminal
- A reply arrive back on WhatsApp (using the placeholder AI response until you wire up a real LLM)

## Next steps

- Replace the placeholder logic in `whatsapp_bot/services.py` → `get_ai_response()` with a real LLM call.
- Create an admin superuser (`python manage.py createsuperuser`) to view Contacts/Messages at `/admin/`.
- Once stable, move off the temporary 24-hour Meta access token to a permanent one via Meta's business verification.
