#!/bin/bash

# AIDB Server Setup Script for Oracle Linux
# This script sets up AIDB on a fresh Oracle Linux instance

set -e

# Configuration
AIDB_USER="aidb"
AIDB_PORT="8080"
DOCKER_REPO="ghcr.io/your-username/aidb"  # Update with your registry

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update system
update_system() {
    echo_info "Updating system packages..."
    sudo dnf update -y
    sudo dnf install -y curl wget git
}

# Install Docker
install_docker() {
    echo_info "Installing Docker..."

    # Add Docker repo
    sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

    # Install Docker
    sudo dnf install -y docker-ce docker-ce-cli containerd.io

    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker

    # Add user to docker group
    sudo usermod -aG docker $USER

    echo_info "Docker installed successfully"
}

# Install Rust (for building from source if needed)
install_rust() {
    echo_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    echo_info "Rust installed successfully"
}

# Setup firewall rules
setup_firewall() {
    echo_info "Setting up firewall rules..."

    # Enable and start firewalld
    sudo systemctl start firewalld
    sudo systemctl enable firewalld

    # Open required ports
    sudo firewall-cmd --permanent --add-port=22/tcp
    sudo firewall-cmd --permanent --add-port=$AIDB_PORT/tcp
    sudo firewall-cmd --reload

    echo_info "Firewall configured successfully"
}

# Create aidb user
create_user() {
    echo_info "Creating AIDB user..."

    if ! id "$AIDB_USER" &>/dev/null; then
        sudo useradd -r -s /bin/false -m -d /opt/aidb $AIDB_USER
        echo_info "User $AIDB_USER created"
    else
        echo_info "User $AIDB_USER already exists"
    fi
}

# Setup AIDB directories
setup_directories() {
    echo_info "Setting up AIDB directories..."

    sudo mkdir -p /opt/aidb/{data,logs,config}
    sudo chown -R $AIDB_USER:$AIDB_USER /opt/aidb

    echo_info "Directories created successfully"
}

# Create systemd service
create_service() {
    echo_info "Creating systemd service..."

    sudo tee /etc/systemd/system/aidb.service > /dev/null <<EOF
[Unit]
Description=AIDB Server
Documentation=https://github.com/your-username/aidb
After=network.target docker.service
Requires=docker.service

[Service]
Type=exec
User=root
Group=root
ExecStartPre=-/usr/bin/docker stop aidb-server
ExecStartPre=-/usr/bin/docker rm aidb-server
ExecStart=/usr/bin/docker run --name aidb-server -p $AIDB_PORT:$AIDB_PORT -v /opt/aidb/data:/app/data $DOCKER_REPO:latest
ExecStop=/usr/bin/docker stop aidb-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo_info "Systemd service created"
}

# Build AIDB from source (alternative to Docker)
build_from_source() {
    echo_info "Building AIDB from source..."

    # Clone repository
    cd /tmp
    git clone https://github.com/your-username/aidb.git
    cd aidb

    # Build release binary
    cargo build --release --package aidb-server

    # Install binary
    sudo cp target/release/aidb-server /usr/local/bin/
    sudo chmod +x /usr/local/bin/aidb-server

    # Create configuration
    sudo tee /opt/aidb/config/config.toml > /dev/null <<EOF
[server]
host = "0.0.0.0"
port = $AIDB_PORT

[storage]
data_dir = "/opt/aidb/data"

[logging]
level = "info"
log_dir = "/opt/aidb/logs"
EOF

    # Update systemd service for binary execution
    sudo tee /etc/systemd/system/aidb.service > /dev/null <<EOF
[Unit]
Description=AIDB Server
Documentation=https://github.com/your-username/aidb
After=network.target

[Service]
Type=simple
User=$AIDB_USER
Group=$AIDB_USER
WorkingDirectory=/opt/aidb
ExecStart=/usr/local/bin/aidb-server --config /opt/aidb/config/config.toml
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo_info "AIDB built and configured successfully"
}

# Install monitoring tools
install_monitoring() {
    echo_info "Installing monitoring tools..."

    # Install htop, iostat, etc.
    sudo dnf install -y htop sysstat net-tools

    echo_info "Monitoring tools installed"
}

# Setup log rotation
setup_logrotate() {
    echo_info "Setting up log rotation..."

    sudo tee /etc/logrotate.d/aidb > /dev/null <<EOF
/opt/aidb/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $AIDB_USER $AIDB_USER
}
EOF

    echo_info "Log rotation configured"
}

# Start services
start_services() {
    echo_info "Starting AIDB service..."

    sudo systemctl enable aidb
    sudo systemctl start aidb

    echo_info "AIDB service started"

    # Show status
    sleep 5
    sudo systemctl status aidb --no-pager
}

# Health check
health_check() {
    echo_info "Performing health check..."

    # Wait a bit for service to start
    sleep 10

    # Get local IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    # Test health endpoint
    if curl -f http://localhost:$AIDB_PORT/health >/dev/null 2>&1; then
        echo_info "✅ AIDB is running successfully!"
        echo_info "🌐 Access AIDB at: http://$LOCAL_IP:$AIDB_PORT"
    else
        echo_warn "⚠️  Health check failed. Check logs:"
        echo "sudo journalctl -u aidb -f"
    fi
}

# Show deployment info
show_info() {
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    echo_info "🎉 AIDB Deployment Complete!"
    echo ""
    echo "📍 Server Details:"
    echo "   - Local IP: $LOCAL_IP"
    echo "   - Port: $AIDB_PORT"
    echo "   - Data Directory: /opt/aidb/data"
    echo "   - Logs Directory: /opt/aidb/logs"
    echo ""
    echo "🔧 Management Commands:"
    echo "   - Start: sudo systemctl start aidb"
    echo "   - Stop: sudo systemctl stop aidb"
    echo "   - Restart: sudo systemctl restart aidb"
    echo "   - Status: sudo systemctl status aidb"
    echo "   - Logs: sudo journalctl -u aidb -f"
    echo ""
    echo "🌐 Access URLs:"
    echo "   - Health: http://$LOCAL_IP:$AIDB_PORT/health"
    echo "   - API: http://$LOCAL_IP:$AIDB_PORT"
    echo ""
}

# Main execution
main() {
    echo_info "Starting AIDB setup on Oracle Linux..."

    update_system
    install_docker
    install_rust
    setup_firewall
    create_user
    setup_directories

    # Choose installation method
    echo_info "Choose installation method:"
    echo "1) Docker (recommended)"
    echo "2) Build from source"
    read -p "Enter choice (1-2): " choice

    case $choice in
        1)
            create_service
            ;;
        2)
            build_from_source
            ;;
        *)
            echo_error "Invalid choice. Using Docker method."
            create_service
            ;;
    esac

    install_monitoring
    setup_logrotate
    start_services
    health_check
    show_info

    echo_info "🚀 AIDB setup complete!"
}

main "$@"