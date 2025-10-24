# Grafana Authentication Troubleshooting Guide

## Problem: Invalid Credentials Error

If you're experiencing "Invalid credentials" when trying to log into Grafana, this guide will help you resolve the issue.

## Root Cause

Grafana environment variables (`GF_SECURITY_ADMIN_USER` and `GF_SECURITY_ADMIN_PASSWORD`) **only work during initial container setup**. Once the Grafana database is created in the persistent volume, these variables are completely ignored, even if you change them in `docker-compose.yml`.

This is **by design**, not a bug. Grafana stores credentials in its SQLite database (`grafana.db`), which persists in the Docker volume.

## Quick Fix Solutions

### Solution 1: Reset Password Using grafana-cli (Preserves Data)

**Best for:** When you want to keep your dashboards and configurations.

```bash
# Reset the admin password
docker exec aegis-grafana grafana cli admin reset-admin-password aegis2024

# Verify it works
curl -u admin:aegis2024 http://localhost:3000/api/health
```

Expected output:
```
Admin password changed successfully ✔
```

---

### Solution 2: Fresh Start (Removes All Data)

**Best for:** New installations or when you don't need existing data.

```bash
# Stop and remove Grafana container
docker-compose stop grafana
docker-compose rm -f grafana

# Remove the persistent volume (THIS DELETES ALL GRAFANA DATA!)
docker volume rm open-agi_grafana-data

# Recreate with fresh credentials from .env file
docker-compose up -d grafana

# Wait 15 seconds for Grafana to initialize
sleep 15

# Verify new credentials work
curl -u admin:aegis2024 http://localhost:3000/api/health
```

⚠️ **Warning:** This deletes ALL Grafana data including dashboards, datasources, and users.

---

### Solution 3: Try Default Credentials

Grafana's default credentials are:
- **Username:** `admin`
- **Password:** `admin`

If you haven't removed the volume since the first deployment, try these:

```bash
curl -u admin:admin http://localhost:3000/api/health
```

If this works, you can change the password in Grafana UI or use grafana-cli.

---

## Step-by-Step Troubleshooting

### Step 1: Check Which Credentials Are Currently Valid

```bash
# Try default Grafana credentials
curl -u admin:admin http://localhost:3000/api/health

# Try credentials from docker-compose.yml
curl -u admin:aegis2024 http://localhost:3000/api/health
```

If either works, you've found the current password!

### Step 2: Check Container Status

```bash
# Verify Grafana is running
docker ps --filter "name=grafana"

# Check logs for errors
docker logs aegis-grafana --tail 50
```

### Step 3: Verify Environment Variables

```bash
# Check if environment variables are set in container
docker exec aegis-grafana env | grep GF_SECURITY
```

Expected output:
```
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=aegis2024
```

If these don't show up, your `.env` file isn't being loaded.

### Step 4: Check if Database Exists

```bash
# Check if grafana.db exists (means credentials are in database)
docker exec aegis-grafana ls -lh /var/lib/grafana/grafana.db
```

If the file exists, environment variables are being ignored.

---

## Using .env File for Credentials

### Create .env file (if not exists)

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred credentials
nano .env
```

Edit these values in `.env`:
```bash
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_password_here
GRAFANA_ROOT_URL=http://localhost:3000
```

### Apply New Credentials (Fresh Install Only)

```bash
# Remove old volume
docker volume rm open-agi_grafana-data

# Start with new credentials from .env
docker-compose up -d grafana
```

---

## Changing Credentials After Installation

### Method 1: Using grafana-cli (Recommended)

```bash
docker exec aegis-grafana grafana cli admin reset-admin-password NewPassword123
```

### Method 2: Using Grafana Web UI

1. Log in with current credentials
2. Go to Profile (bottom left) → Preferences
3. Click "Change Password"
4. Enter current and new password

### Method 3: Using Grafana API

```bash
# Get user ID (usually 1 for admin)
curl -u admin:current_password http://localhost:3000/api/users/lookup?loginOrEmail=admin

# Change password
curl -X PUT -H "Content-Type: application/json" \
  -u admin:current_password \
  -d '{"password":"new_password"}' \
  http://localhost:3000/api/admin/users/1/password
```

---

## Production Best Practices

### 1. Never Hardcode Credentials

❌ **Don't do this:**
```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=aegis2024
```

✅ **Do this instead:**
```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-aegis2024}
```

### 2. Don't Commit .env Files

Add to `.gitignore`:
```
.env
.env.local
.env.production
```

Keep `.env.example` in git as a template.

### 3. Use Strong Passwords in Production

Default password `aegis2024` is only for development. In production:
- Use passwords with 16+ characters
- Include uppercase, lowercase, numbers, and symbols
- Use a password manager
- Rotate credentials regularly

### 4. Consider OAuth/LDAP

For production deployments, consider external authentication:
- GitHub OAuth
- Google OAuth
- LDAP/Active Directory
- SAML

Configure in `grafana.ini` or environment variables.

---

## Common Errors and Solutions

### Error: "Invalid username or password"

**Cause:** Credentials don't match what's in the database.

**Solution:** Use Solution 1 (grafana-cli reset) or Solution 2 (fresh start).

---

### Error: "Client sent an HTTP request to an HTTPS server"

**Cause:** Accessing Grafana with wrong protocol.

**Solution:** Use `http://localhost:3000` not `https://localhost:3000`.

---

### Error: "EOF" from curl

**Cause:** Grafana container not running or not ready.

**Solution:**
```bash
# Check container status
docker ps --filter "name=grafana"

# Wait for Grafana to start
docker logs -f aegis-grafana
```

Wait for log line: `HTTP Server Listen`

---

### Environment Variables Not Working

**Cause:** Variables only work on first initialization.

**Solution:** Remove volume and restart (see Solution 2).

---

## Quick Reference Commands

```bash
# Reset password (keeps data)
docker exec aegis-grafana grafana cli admin reset-admin-password aegis2024

# Fresh install (deletes data)
docker volume rm open-agi_grafana-data && docker-compose up -d grafana

# Check credentials work
curl -u admin:aegis2024 http://localhost:3000/api/health

# View logs
docker logs aegis-grafana --tail 50

# Access Grafana shell
docker exec -it aegis-grafana /bin/bash
```

---

## Getting Help

If none of these solutions work:

1. Check Grafana logs: `docker logs aegis-grafana`
2. Verify container is healthy: `docker ps --filter "name=grafana"`
3. Check network connectivity: `curl http://localhost:3000`
4. Review Grafana documentation: https://grafana.com/docs/grafana/latest/

---

## Default Credentials (Development Only)

**After fresh installation with current configuration:**
- **Username:** `admin`
- **Password:** `aegis2024`
- **URL:** http://localhost:3000

**⚠️ Change these immediately in production!**
