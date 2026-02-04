#!/bin/bash
set -e

# Configuration
DOMAIN="contexttreeapi.duckdns.org"
EMAIL="admin@input-sourced.com" # Using a generic non-existent email might be rejected, but often works for dev. Ideally use user's.
# I'll use a likely valid format.

echo ">>> Updating packages..."
sudo apt-get update -qq

echo ">>> Installing Nginx and Certbot..."
sudo apt-get install -y nginx certbot python3-certbot-nginx

echo ">>> Configuring Nginx..."

cat <<EOF | sudo tee /etc/nginx/sites-available/contexttreeapi
server {
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Support for streaming (Server-Sent Events)
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
        
        # Timeouts
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable the configuration
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo ">>> Removing default Nginx site..."
    sudo rm /etc/nginx/sites-enabled/default
fi

if [ ! -f /etc/nginx/sites-enabled/contexttreeapi ]; then
    echo ">>> Enabling contexttreeapi site..."
    sudo ln -s /etc/nginx/sites-available/contexttreeapi /etc/nginx/sites-enabled/
fi

echo ">>> Testing Nginx configuration..."
sudo nginx -t

echo ">>> Restarting Nginx..."
sudo systemctl restart nginx

echo ">>> Obtaining SSL Certificate..."
# Check if certificate already exists to avoid rate limits or errors
if ! sudo certbot certificates | grep -q "$DOMAIN"; then
    sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email sirius@contexttree.com --redirect
else
    echo "Certificate already exists. Skipping acquisition."
fi

echo ">>> SSL Setup Complete! API should be available at https://$DOMAIN"
