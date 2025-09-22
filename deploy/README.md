# AIDB Oracle Cloud Deployment Guide

This guide helps you deploy AIDB on Oracle Cloud Infrastructure's Always Free tier.

## Prerequisites

1. **Oracle Cloud Account**: Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. **Oracle CLI**: Install the Oracle CLI tool
3. **Docker**: For containerized deployment
4. **SSH Key**: Generate an SSH key pair

## Oracle Cloud Always Free Resources

Your free tier includes:
- **2 AMD-based VMs**: 1/8 OCPU, 1GB RAM each
- **4 ARM-based VMs**: Up to 4 OCPUs, 24GB RAM total
- **200GB Block Volume storage**
- **Unlimited bandwidth** (ingress/egress)

## Quick Start

### 1. Setup Oracle CLI

```bash
# Install Oracle CLI
curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh | bash

# Configure CLI (requires tenancy OCID, user OCID, etc.)
oci setup config
```

### 2. Get Required OCIDs

From Oracle Cloud Console, note down:
- **Compartment OCID**: Usually your root compartment
- **Availability Domain**: e.g., `Uocm:US-ASHBURN-AD-1`

### 3. Deploy AIDB

```bash
# Make scripts executable
chmod +x deploy/oracle-cloud-setup.sh
chmod +x deploy/setup-aidb.sh

# Deploy to Oracle Cloud
./deploy/oracle-cloud-setup.sh \
  --compartment-id "ocid1.compartment.oc1..your_compartment_id" \
  --availability-domain "Uocm:US-ASHBURN-AD-1"
```

## Manual Deployment Steps

### 1. Create Compute Instance

1. Go to Oracle Cloud Console → Compute → Instances
2. Click "Create Instance"
3. Configure:
   - **Name**: aidb-server
   - **Image**: Oracle Linux 8
   - **Shape**: VM.Standard.E2.1.Micro (Always Free)
   - **Add SSH Key**: Upload your public key
4. Click "Create"

### 2. Configure Security Rules

1. Go to Networking → Virtual Cloud Networks
2. Select your VCN → Security Lists
3. Add ingress rules:
   - **Port 22**: SSH access (0.0.0.0/0)
   - **Port 8080**: AIDB server (0.0.0.0/0)

### 3. Setup AIDB on Instance

```bash
# SSH to your instance
ssh opc@<instance-public-ip>

# Upload and run setup script
wget https://raw.githubusercontent.com/your-username/aidb/main/deploy/setup-aidb.sh
chmod +x setup-aidb.sh
./setup-aidb.sh
```

## Docker Deployment

### Build and Push Image

```bash
# Build Docker image
docker build -t aidb:latest .

# Tag for registry
docker tag aidb:latest ghcr.io/your-username/aidb:latest

# Push to registry
docker push ghcr.io/your-username/aidb:latest
```

### Run on Oracle Cloud

```bash
# On your Oracle Cloud instance
docker run -d \
  --name aidb-server \
  -p 8080:8080 \
  -v /opt/aidb/data:/app/data \
  ghcr.io/your-username/aidb:latest
```

## Configuration

### Environment Variables

```bash
export RUST_LOG=info
export AIDB_HOST=0.0.0.0
export AIDB_PORT=8080
export AIDB_DATA_DIR=/opt/aidb/data
```

### Config File (`/opt/aidb/config/config.toml`)

```toml
[server]
host = "0.0.0.0"
port = 8080

[storage]
data_dir = "/opt/aidb/data"

[logging]
level = "info"
log_dir = "/opt/aidb/logs"
```

## Management Commands

```bash
# Service management
sudo systemctl start aidb
sudo systemctl stop aidb
sudo systemctl restart aidb
sudo systemctl status aidb

# View logs
sudo journalctl -u aidb -f

# Health check
curl http://localhost:8080/health
```

## Monitoring

### System Resources

```bash
# Check CPU/memory usage
htop

# Check disk usage
df -h

# Check network connections
ss -tulpn | grep 8080
```

### AIDB Metrics

Access monitoring at: `http://your-instance-ip:8080/metrics`

## Troubleshooting

### Common Issues

1. **"Out of host capacity"**: Oracle free tier capacity exhausted
   - Wait and retry later
   - Try different availability domain

2. **Connection refused**:
   - Check firewall rules
   - Verify service is running: `sudo systemctl status aidb`

3. **Build failures**:
   - Ensure adequate memory (may need swap on 1GB instances)
   - Use Docker for consistent builds

### Instance Performance Tips

For 1GB RAM instances:
```bash
# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Security Considerations

1. **Change default SSH port**:
   ```bash
   sudo sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
   sudo systemctl restart sshd
   ```

2. **Setup fail2ban**:
   ```bash
   sudo dnf install epel-release -y
   sudo dnf install fail2ban -y
   sudo systemctl enable fail2ban
   ```

3. **Regular updates**:
   ```bash
   sudo dnf update -y
   ```

## Costs and Limits

- **Always Free resources**: No charges, ever
- **Idle policy**: Instances with <20% CPU usage for 7 days may be reclaimed
- **Resource limits**: Cannot exceed Always Free allocations

## Support

- **Oracle Cloud Documentation**: [docs.oracle.com](https://docs.oracle.com/en-us/iaas/)
- **AIDB Issues**: Create issues in the GitHub repository
- **Community**: Oracle Cloud forums and Discord

## Next Steps

1. **Custom Domain**: Setup DNS and SSL certificate
2. **Load Balancer**: Use Oracle Cloud Load Balancer for HA
3. **Backup Strategy**: Implement automated backups
4. **Monitoring**: Add Prometheus/Grafana stack
5. **CI/CD**: Setup automated deployments