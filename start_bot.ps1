# One-click demo startup -- opens Django, Celery, and ngrok each in their own
# window, from the venv, in the right order. Run this instead of typing the
# three commands by hand every time.
#
# Before first use: replace YOUR-STATIC-DOMAIN below with the static domain
# you reserved at https://dashboard.ngrok.com/domains (Domains -> Create Domain).
# If you haven't reserved one yet, leave $NgrokDomain blank -- it'll fall back
# to a random URL (fine for testing, but you'll need to re-paste the webhook
# URL into Meta every time you restart ngrok).

$ProjectRoot = $PSScriptRoot
$NgrokDomain = ""   # e.g. "taxplan-demo.ngrok-free.app"

# Kill any runserver/celery worker/celery beat left over from a previous run
# of this script (or from manually-typed commands in another window) before
# launching fresh ones. Without this, re-running the script just piles up
# duplicate processes -- including ones started from the wrong Python (e.g.
# anaconda3 instead of venv, missing this project's dependencies) -- and only
# one of them ends up actually bound to port 8000, so it's a coin flip which
# environment is really answering webhook traffic.
Write-Host "Stopping any leftover runserver/celery/ngrok processes from a previous run..." -ForegroundColor Cyan
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'manage\.py runserver' -or $_.CommandLine -match 'celery -A myproject' -or $_.CommandLine -match '-m celery' } |
    ForEach-Object {
        Write-Host "  killing PID $($_.ProcessId): $($_.CommandLine)" -ForegroundColor DarkGray
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
# ngrok too -- a second tunnel means a second, different public URL, and only
# one of them can possibly match what's saved in Meta's webhook config.
Get-Process ngrok -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  killing ngrok PID $($_.Id)" -ForegroundColor DarkGray
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

Write-Host "Starting Django, Celery, and ngrok in separate windows..." -ForegroundColor Cyan

# Django dev server
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; venv\Scripts\Activate.ps1; python manage.py runserver"
)

# Celery worker (--pool=solo is required on Windows)
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; venv\Scripts\Activate.ps1; celery -A myproject worker -l info --pool=solo"
)

# Celery Beat -- drives the periodic abandoned-cart nudge task (see
# myproject/celery.py's beat_schedule). Without this window running,
# send_abandoned_cart_nudges silently never fires; everything else (replies,
# CSAT, callback bookings, the referral ask) still works fine without it.
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; venv\Scripts\Activate.ps1; celery -A myproject beat -l info"
)

# ngrok tunnel
if ($NgrokDomain -ne "") {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$ProjectRoot'; ngrok http --domain=$NgrokDomain 8000"
    )
} else {
    Write-Host "No static ngrok domain set in this script -- using a random URL." -ForegroundColor Yellow
    Write-Host "You'll need to copy the new https URL into Meta's webhook config." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$ProjectRoot'; ngrok http 8000"
    )
}

Write-Host ""
Write-Host "All three windows launching. Give them ~5-10 seconds to come up," -ForegroundColor Cyan
Write-Host "then run .\check_demo_ready.ps1 to confirm everything's actually alive." -ForegroundColor Cyan
