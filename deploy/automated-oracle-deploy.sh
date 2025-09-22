#!/bin/bash

# Automated Oracle Cloud AIDB Deployment Script
# This script provides full automation for Oracle Cloud deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_NAME="aidb-server"
IMAGE_ID="ocid1.image.oc1.iad.aaaaaaaag2uyozo7266bmg26j5rrivi62spram6sps2u3ikbt3c3yrmqmeaa"  # Oracle Linux 8 AMD64
SHAPE="VM.Standard.A1.Flex"  # ARM instance for trial credits
SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"

# Ensure Oracle CLI is in PATH
export PATH="/Users/vivekkumar/Library/Python/3.9/bin:$PATH"

# Banner
echo "=================================="
echo_info "🚀 AIDB Oracle Cloud Deployer"
echo_info "Automated deployment with trial credits"
echo "=================================="
echo

# Step 1: Check prerequisites
echo_step "1/8 Checking prerequisites..."

if ! command -v oci &> /dev/null; then
    echo_error "Oracle CLI not found in PATH"
    echo "Please ensure Oracle CLI is installed and in PATH:"
    echo "export PATH=\"/Users/vivekkumar/Library/Python/3.9/bin:\$PATH\""
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo_error "Docker not found. Please install Docker first."
    exit 1
fi

if [ ! -f "$SSH_KEY_PATH" ]; then
    echo_error "SSH public key not found at $SSH_KEY_PATH"
    echo "Generate one with: ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'"
    exit 1
fi

echo_info "✅ Prerequisites check passed!"
echo

# Step 2: Configure Oracle CLI (if not already configured)
echo_step "2/8 Configuring Oracle CLI..."

if [ ! -f "$HOME/.oci/config" ]; then
    echo_warn "Oracle CLI not configured. Starting configuration..."
    echo_info "You'll need:"
    echo "  - Tenancy OCID (from Oracle Cloud Console)"
    echo "  - User OCID (from Oracle Cloud Console)"
    echo "  - Region (e.g., us-ashburn-1)"
    echo "  - API Key fingerprint"
    echo "  - Path to private key"
    echo
    oci setup config
else
    echo_info "✅ Oracle CLI already configured"
fi
echo

# Step 3: Get required OCIDs
echo_step "3/8 Getting Oracle Cloud information..."

# Prompt for compartment ID if not provided
if [ -z "$COMPARTMENT_ID" ]; then
    echo_info "Getting your compartment information..."
    echo "Available compartments:"
    oci iam compartment list --query 'data[*].{Name:name,OCID:id}' --output table 2>/dev/null || true
    echo
    read -p "Enter your compartment OCID (usually root compartment): " COMPARTMENT_ID
fi

# Prompt for availability domain if not provided
if [ -z "$AVAILABILITY_DOMAIN" ]; then
    echo_info "Getting availability domains..."
    echo "Available domains:"
    oci iam availability-domain list --compartment-id "$COMPARTMENT_ID" --query 'data[*].name' --output table 2>/dev/null || true
    echo
    read -p "Enter availability domain (e.g., Uocm:US-ASHBURN-AD-1): " AVAILABILITY_DOMAIN
fi

echo_info "✅ Configuration collected"
echo

# Step 4: Create VCN and networking
echo_step "4/8 Setting up network infrastructure..."

# Check if VCN exists
VCN_ID=$(oci network vcn list --compartment-id "$COMPARTMENT_ID" --display-name "aidb-vcn" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

if [ "$VCN_ID" = "null" ]; then
    echo_info "Creating VCN..."
    VCN_ID=$(oci network vcn create \
        --compartment-id "$COMPARTMENT_ID" \
        --display-name "aidb-vcn" \
        --cidr-block "10.0.0.0/16" \
        --query 'data.id' --raw-output)
    echo_info "VCN created: $VCN_ID"

    # Wait for VCN to be available
    echo_info "Waiting for VCN to be available..."
    sleep 10
else
    echo_info "Using existing VCN: $VCN_ID"
fi

# Create internet gateway
IG_ID=$(oci network internet-gateway list --compartment-id "$COMPARTMENT_ID" --vcn-id "$VCN_ID" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

if [ "$IG_ID" = "null" ]; then
    echo_info "Creating Internet Gateway..."
    IG_ID=$(oci network internet-gateway create \
        --compartment-id "$COMPARTMENT_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "aidb-ig" \
        --is-enabled true \
        --query 'data.id' --raw-output)
    echo_info "Internet Gateway created: $IG_ID"

    # Add route rule for internet gateway
    echo_info "Adding route rule..."
    ROUTE_TABLE_ID=$(oci network vcn get --vcn-id "$VCN_ID" --query 'data."default-route-table-id"' --raw-output)
    oci network route-table update \
        --rt-id "$ROUTE_TABLE_ID" \
        --route-rules '[{"destination": "0.0.0.0/0", "destinationType": "CIDR_BLOCK", "networkEntityId": "'$IG_ID'"}]' \
        --force > /dev/null
else
    echo_info "Using existing Internet Gateway: $IG_ID"
fi

# Create subnet
SUBNET_ID=$(oci network subnet list --compartment-id "$COMPARTMENT_ID" --vcn-id "$VCN_ID" --display-name "aidb-subnet" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

if [ "$SUBNET_ID" = "null" ]; then
    echo_info "Creating subnet..."
    SUBNET_ID=$(oci network subnet create \
        --compartment-id "$COMPARTMENT_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "aidb-subnet" \
        --cidr-block "10.0.1.0/24" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --query 'data.id' --raw-output)
    echo_info "Subnet created: $SUBNET_ID"
else
    echo_info "Using existing subnet: $SUBNET_ID"
fi

echo_info "✅ Network infrastructure ready"
echo

# Step 5: Create security list with required ports
echo_step "5/8 Configuring security rules..."

SECURITY_LIST_ID=$(oci network security-list list --compartment-id "$COMPARTMENT_ID" --vcn-id "$VCN_ID" --display-name "aidb-security-list" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

if [ "$SECURITY_LIST_ID" = "null" ]; then
    echo_info "Creating security list..."
    SECURITY_LIST_ID=$(oci network security-list create \
        --compartment-id "$COMPARTMENT_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "aidb-security-list" \
        --ingress-security-rules '[
            {
                "source": "0.0.0.0/0",
                "protocol": "6",
                "tcpOptions": {
                    "destinationPortRange": {
                        "min": 22,
                        "max": 22
                    }
                }
            },
            {
                "source": "0.0.0.0/0",
                "protocol": "6",
                "tcpOptions": {
                    "destinationPortRange": {
                        "min": 8080,
                        "max": 8080
                    }
                }
            }
        ]' \
        --egress-security-rules '[
            {
                "destination": "0.0.0.0/0",
                "protocol": "all"
            }
        ]' \
        --query 'data.id' --raw-output)
    echo_info "Security list created: $SECURITY_LIST_ID"

    # Associate security list with subnet
    echo_info "Associating security list with subnet..."
    oci network subnet update \
        --subnet-id "$SUBNET_ID" \
        --security-list-ids "[\"$SECURITY_LIST_ID\"]" \
        --force > /dev/null
else
    echo_info "Using existing security list: $SECURITY_LIST_ID"
fi

echo_info "✅ Security rules configured"
echo

# Step 6: Launch compute instance
echo_step "6/8 Launching compute instance..."

# Check if instance already exists
EXISTING_INSTANCE=$(oci compute instance list --compartment-id "$COMPARTMENT_ID" --display-name "$INSTANCE_NAME" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

if [ "$EXISTING_INSTANCE" != "null" ]; then
    echo_warn "Instance $INSTANCE_NAME already exists: $EXISTING_INSTANCE"
    INSTANCE_ID="$EXISTING_INSTANCE"
else
    echo_info "Creating ARM instance with trial credits..."
    echo_info "Shape: $SHAPE (2 cores, 12GB RAM)"

    INSTANCE_ID=$(oci compute instance launch \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --shape "$SHAPE" \
        --shape-config '{"memoryInGBs": 12, "ocpus": 2}' \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --display-name "$INSTANCE_NAME" \
        --ssh-authorized-keys-file "$SSH_KEY_PATH" \
        --assign-public-ip true \
        --wait-for-state RUNNING \
        --query 'data.id' --raw-output)

    echo_info "✅ Instance launched successfully: $INSTANCE_ID"
fi

# Get public IP
echo_info "Getting instance details..."
PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output)
echo_info "Public IP: $PUBLIC_IP"

echo_info "✅ Instance ready"
echo

# Step 7: Wait for SSH and deploy AIDB
echo_step "7/8 Deploying AIDB to instance..."

echo_info "Waiting for SSH to be available..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no opc@$PUBLIC_IP "echo 'SSH ready'" 2>/dev/null; then
        echo_info "✅ SSH connection established"
        break
    fi
    echo_info "Attempt $attempt/$max_attempts - waiting for SSH..."
    sleep 10
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo_error "Could not establish SSH connection after $max_attempts attempts"
    exit 1
fi

# Upload and run deployment script
echo_info "Uploading AIDB setup script..."
scp -o StrictHostKeyChecking=no "$SCRIPT_DIR/setup-aidb.sh" opc@$PUBLIC_IP:~/

echo_info "Running AIDB installation..."
ssh -o StrictHostKeyChecking=no opc@$PUBLIC_IP 'chmod +x setup-aidb.sh && echo "1" | ./setup-aidb.sh'

echo_info "✅ AIDB deployed successfully"
echo

# Step 8: Final verification and information
echo_step "8/8 Final verification..."

echo_info "Testing AIDB health endpoint..."
sleep 10

if curl -f "http://$PUBLIC_IP:8080/health" >/dev/null 2>&1; then
    echo_info "✅ AIDB is running successfully!"
else
    echo_warn "⚠️  Health check failed. AIDB may still be starting up."
    echo_info "Check status with: ssh opc@$PUBLIC_IP 'sudo systemctl status aidb'"
fi

echo
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=================================="
echo_info "📍 Instance Details:"
echo "   - Name: $INSTANCE_NAME"
echo "   - Shape: $SHAPE (2 cores, 12GB RAM)"
echo "   - Public IP: $PUBLIC_IP"
echo "   - Cost: ~\$6/month with trial credits"
echo ""
echo_info "🌐 Access URLs:"
echo "   - AIDB Server: http://$PUBLIC_IP:8080"
echo "   - Health Check: http://$PUBLIC_IP:8080/health"
echo "   - SSH Access: ssh opc@$PUBLIC_IP"
echo ""
echo_info "🔧 Management Commands:"
echo "   - Status: ssh opc@$PUBLIC_IP 'sudo systemctl status aidb'"
echo "   - Logs: ssh opc@$PUBLIC_IP 'sudo journalctl -u aidb -f'"
echo "   - Restart: ssh opc@$PUBLIC_IP 'sudo systemctl restart aidb'"
echo ""
echo_info "💰 Trial Credits:"
echo "   - Monthly cost: ~\$6 (from your \$300 trial credits)"
echo "   - Credits remaining: Check Oracle Cloud Console"
echo "=================================="

# Save deployment info for future reference
cat > "$SCRIPT_DIR/deployment-info.txt" << EOF
AIDB Oracle Cloud Deployment Information
========================================
Deployment Date: $(date)
Instance ID: $INSTANCE_ID
Instance Name: $INSTANCE_NAME
Public IP: $PUBLIC_IP
Shape: $SHAPE
Compartment ID: $COMPARTMENT_ID
VCN ID: $VCN_ID
Subnet ID: $SUBNET_ID

Access Information:
- AIDB URL: http://$PUBLIC_IP:8080
- SSH: ssh opc@$PUBLIC_IP
- Health: http://$PUBLIC_IP:8080/health

Management:
- Status: ssh opc@$PUBLIC_IP 'sudo systemctl status aidb'
- Logs: ssh opc@$PUBLIC_IP 'sudo journalctl -u aidb -f'
- Restart: ssh opc@$PUBLIC_IP 'sudo systemctl restart aidb'
EOF

echo_info "📝 Deployment information saved to: $SCRIPT_DIR/deployment-info.txt"
echo_info "🚀 AIDB is now ready for use!"