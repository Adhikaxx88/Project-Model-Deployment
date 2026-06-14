

set -euo pipefail


GIT_REPO="https://github.com/Adhikaxx88/Project-Model-Deployment.git"
SUBFOLDER="2/2B"
ENDPOINT_NAME="credit-score-endpoint"
AWS_REGION="us-east-1"


APP_DIR=/opt/credit-score-app

dnf update -y
dnf install -y git python3 python3-pip

git clone "$GIT_REPO" /tmp/repo
mkdir -p "$APP_DIR"
if [ -n "$SUBFOLDER" ]; then
    cp -r /tmp/repo/"$SUBFOLDER"/. "$APP_DIR/"
else
    cp -r /tmp/repo/. "$APP_DIR/"
fi
rm -rf /tmp/repo

pip3 install -r "$APP_DIR/requirements.txt"

# FastAPI service (port 8000)
cat > /etc/systemd/system/fastapi.service << EOF
[Unit]
Description=Credit Score FastAPI Cloud
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
Environment="ENDPOINT_NAME=$ENDPOINT_NAME"
Environment="AWS_REGION=$AWS_REGION"
ExecStart=/usr/bin/python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Streamlit service (port 8501)
cat > /etc/systemd/system/streamlit.service << EOF
[Unit]
Description=Credit Score Streamlit
After=network.target fastapi.service

[Service]
User=root
WorkingDirectory=$APP_DIR
Environment="ENDPOINT_NAME=$ENDPOINT_NAME"
Environment="AWS_REGION=$AWS_REGION"
ExecStart=/usr/bin/python3 -m streamlit run frontend.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.serverAddress 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable fastapi streamlit
systemctl start fastapi streamlit

touch "$APP_DIR/.userdata-success"
echo "Bootstrap selesai "
