# Run this right before a demo. Checks the four things that have actually
# broken this bot before: Django up, Celery worker alive, ngrok tunnel up,
# WhatsApp token still valid. Takes a few seconds, saves you from finding out
# mid-demo.

$ProjectRoot = $PSScriptRoot
$AllGood = $true

function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Bad($msg)  { Write-Host "  [FAIL] $msg" -ForegroundColor Red; $script:AllGood = $false }
function Info($msg) { Write-Host "  $msg" -ForegroundColor DarkGray }

Write-Host "`n1. Django (port 8000)" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/admin/login/" -UseBasicParsing -TimeoutSec 5
    Ok "responding (HTTP $($r.StatusCode))"
} catch {
    Bad "not responding on port 8000 — is `"python manage.py runserver`" running?"
}

Write-Host "`n2. Celery worker" -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    $celeryOut = & venv\Scripts\celery.exe -A myproject inspect ping 2>&1 | Out-String
    if ($celeryOut -match "pong") {
        Ok "worker responded to ping"
    } else {
        Bad "no worker replied — is `"celery -A myproject worker -l info --pool=solo`" running?"
        Info ($celeryOut.Trim())
    }
} catch {
    Bad "couldn't run celery inspect — check venv path"
}
Pop-Location

Write-Host "`n3. ngrok tunnel" -ForegroundColor Cyan
try {
    $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5
    $httpsTunnel = $tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
    if ($httpsTunnel) {
        Ok "tunnel live: $($httpsTunnel.public_url) -> $($httpsTunnel.config.addr)"
        if ($httpsTunnel.config.addr -notmatch ":8000$") {
            Bad "tunnel points to $($httpsTunnel.config.addr), not port 8000 — this is how the 502 happened before"
        }
    } else {
        Bad "ngrok running but no https tunnel found"
    }
} catch {
    Bad "ngrok inspector not reachable on 4040 — is ngrok running?"
}

Write-Host "`n4. WhatsApp access token" -ForegroundColor Cyan
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
    $token = if ($envContent -match "WHATSAPP_ACCESS_TOKEN=(.+)") { $matches[1].Trim() } else { $null }
    $phoneId = if ($envContent -match "WHATSAPP_PHONE_ID=(.+)") { $matches[1].Trim() } else { $null }
    if ($token -and $phoneId) {
        try {
            $resp = Invoke-RestMethod -Uri "https://graph.facebook.com/v22.0/$phoneId`?access_token=$token" -TimeoutSec 10
            Ok "token valid (phone: $($resp.display_phone_number))"
        } catch {
            Bad "token rejected by Meta — it's expired or malformed. Get a fresh one."
        }
    } else {
        Bad "WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_ID missing from .env"
    }
} else {
    Bad ".env not found at $envFile"
}

Write-Host ""
if ($AllGood) {
    Write-Host "All checks passed — good to demo." -ForegroundColor Green
} else {
    Write-Host "Something needs fixing before you demo — see FAIL lines above." -ForegroundColor Red
}
