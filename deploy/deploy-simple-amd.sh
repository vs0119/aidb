#!/bin/bash

# Simple AMD Always Free Instance Launcher
# Uses existing infrastructure to launch VM.Standard.E2.1.Micro

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
TENANCY_ID="ocid1.tenancy.oc1..aaaaaaaap7ospwxuoqjhqkf4ec3mpz7o3q3nue6kfcc3c7lxu4zkh3b2ophq"
INSTANCE_NAME="aidb-server"
SUBNET_ID="ocid1.subnet.oc1.us-chicago-1.aaaaaaaaasmip4jq6pi5wcgpxo2ixka7dakcgd6qpn3qsnyxxt2665htiosa"
AMD_IMAGE_ID="ocid1.image.oc1.us-chicago-1.aaaaaaaaeae3ejm7ismgu6ebhuk7rdbm6a26dqzjwxfxnooybg5iyo2vqdhq"
AVAILABILITY_DOMAINS=("GtHP:US-CHICAGO-1-AD-1" "GtHP:US-CHICAGO-1-AD-2" "GtHP:US-CHICAGO-1-AD-3")

# Ensure Oracle CLI is in PATH
export PATH="/Users/vivekkumar/Library/Python/3.9/bin:$PATH"
export OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True
export SUPPRESS_LABEL_WARNING=True

echo "=================================================="
echo_info "🚀 Simple AMD Always Free Instance Launcher"
echo_info "Using existing infrastructure"
echo_info "VM.Standard.E2.1.Micro (1/8 OCPU, 1GB RAM)"
echo "=================================================="
echo

# Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo_error "SSH public key not found at ~/.ssh/id_rsa.pub"
    exit 1
fi

echo_info "Using SSH key: $(cat ~/.ssh/id_rsa.pub | cut -d' ' -f1-2)"
echo_info "Using subnet: $SUBNET_ID"
echo_info "Using image: $AMD_IMAGE_ID"
echo

# Try each availability domain with extended retry
INSTANCE_ID=""
for ad in "${AVAILABILITY_DOMAINS[@]}"; do
    echo_step "Trying to launch in $ad..."

    # Try multiple times for this AD
    for attempt in {1..3}; do
        echo_info "Attempt $attempt/3 for $ad"

        if INSTANCE_ID=$(oci compute instance launch \
            --compartment-id "$TENANCY_ID" \
            --availability-domain "$ad" \
            --shape "VM.Standard.E2.1.Micro" \
            --image-id "$AMD_IMAGE_ID" \
            --subnet-id "$SUBNET_ID" \
            --display-name "$INSTANCE_NAME" \
            --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
            --assign-public-ip true \
            --query 'data.id' \
            --raw-output 2>/dev/null); then
            echo_info "✅ Instance launched successfully in $ad!"
            break 2
        else
            echo_warn "Attempt $attempt failed in $ad"
            if [ $attempt -lt 3 ]; then
                echo_info "Waiting 30 seconds before retry..."
                sleep 30
            fi
        fi
    done

    if [ -n "$INSTANCE_ID" ]; then
        break
    fi
done

if [ -z "$INSTANCE_ID" ]; then
    echo_error "❌ Could not launch instance in any availability domain after multiple attempts"
    echo_info "💡 You can try again later or use the Oracle Cloud Console to launch manually"
    echo_info "   Use these settings:"
    echo_info "   - Shape: VM.Standard.E2.1.Micro"
    echo_info "   - Image: Oracle Linux 8"
    echo_info "   - Subnet: aidb-subnet"
    echo_info "   - Assign public IP: Yes"
    exit 1
fi

echo_step "Waiting for instance to be running..."
oci compute instance get \
    --instance-id "$INSTANCE_ID" \
    --wait-for-state RUNNING >/dev/null

echo_step "Getting instance details..."
PUBLIC_IP=$(oci compute instance list-vnics \
    --instance-id "$INSTANCE_ID" \
    --query 'data[0]."public-ip"' \
    --raw-output)

echo_info "Instance ID: $INSTANCE_ID"
echo_info "Public IP: $PUBLIC_IP"

# Save deployment info
cat > "deployment-info.txt" << EOF
AIDB Oracle Cloud Always Free Deployment
========================================
Deployment Date: $(date)
Instance ID: $INSTANCE_ID
Instance Name: $INSTANCE_NAME
Public IP: $PUBLIC_IP
Shape: VM.Standard.E2.1.Micro (Always Free)

Access Information:
- SSH: ssh opc@$PUBLIC_IP
- AIDB will be at: http://$PUBLIC_IP:8080 (after setup)

Setup Commands:
ssh opc@$PUBLIC_IP
# Then run AIDB setup commands
EOF

echo ""
echo "🎉 AMD ALWAYS FREE INSTANCE DEPLOYED!"
echo "====================================="
echo_info "📍 Instance Details:"
echo "   - Shape: VM.Standard.E2.1.Micro (Always Free)"
echo "   - CPU: 1/8 OCPU (~0.125 vCPU)"
echo "   - Memory: 1GB RAM"
echo "   - Cost: FREE FOREVER 💰"
echo ""
echo_info "🌐 Access Information:"
echo "   - SSH: ssh opc@$PUBLIC_IP"
echo "   - Instance ready for AIDB setup"
echo ""
echo_info "📝 Next Steps:"
echo "   1. SSH to the instance: ssh opc@$PUBLIC_IP"
echo "   2. Upload and run AIDB setup script"
echo "   3. Access AIDB at: http://$PUBLIC_IP:8080"
echo "====================================="