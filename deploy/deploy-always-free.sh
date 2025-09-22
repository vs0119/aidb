#!/bin/bash

# Oracle Cloud Always Free AMD Instance Deployment
# Creates VM.Standard.E2.1.Micro (1/8 OCPU, 1GB RAM) - Free Forever

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
TENANCY_ID="ocid1.tenancy.oc1..aaaaaaaap7ospwxuoqjhqkf4ec3mpz7o3q3nue6kfcc3c7lxu4zkh3b2ophq"
INSTANCE_NAME="aidb-server"
VCN_NAME="aidb-vcn"
SUBNET_NAME="aidb-subnet"
SECURITY_LIST_NAME="aidb-security-list"
IG_NAME="aidb-ig"
AVAILABILITY_DOMAINS=("GtHP:US-CHICAGO-1-AD-1" "GtHP:US-CHICAGO-1-AD-2" "GtHP:US-CHICAGO-1-AD-3")

# Ensure Oracle CLI is in PATH
export PATH="/Users/vivekkumar/Library/Python/3.9/bin:$PATH"
export OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True
export SUPPRESS_LABEL_WARNING=True

echo "=================================================="
echo_info "🚀 Oracle Cloud Always Free AIDB Deployment"
echo_info "Creating VM.Standard.E2.1.Micro (1/8 OCPU, 1GB RAM)"
echo_info "💰 Free Forever - No charges ever!"
echo "=================================================="
echo

# Get AMD image ID
AMD_IMAGE_ID=$(oci compute image list \
    --compartment-id "$TENANCY_ID" \
    --operating-system "Oracle Linux" \
    --operating-system-version "8" \
    --query 'data[?!contains("display-name", `aarch64`) && !contains("display-name", `GPU`) && contains("display-name", `2025.08.31`)].id | [0]' \
    --raw-output 2>/dev/null || echo "ocid1.image.oc1.us-chicago-1.aaaaaaaaeae3ejm7ismgu6ebhuk7rdbm6a26dqzjwxfxnooybg5iyo2vqdhq")

echo_info "Using Oracle Linux 8 AMD image: $AMD_IMAGE_ID"

# Step 1: Create VCN
echo_step "1/6 Creating Virtual Cloud Network..."
VCN_ID=$(oci network vcn create \
    --compartment-id "$TENANCY_ID" \
    --display-name "$VCN_NAME" \
    --cidr-block "10.0.0.0/16" \
    --query 'data.id' \
    --raw-output)
echo_info "VCN created: $VCN_ID"
sleep 5

# Step 2: Create Internet Gateway
echo_step "2/6 Creating Internet Gateway..."
IG_ID=$(oci network internet-gateway create \
    --compartment-id "$TENANCY_ID" \
    --vcn-id "$VCN_ID" \
    --display-name "$IG_NAME" \
    --is-enabled true \
    --query 'data.id' \
    --raw-output)
echo_info "Internet Gateway created: $IG_ID"

# Add route rule
ROUTE_TABLE_ID=$(oci network vcn get \
    --vcn-id "$VCN_ID" \
    --query 'data."default-route-table-id"' \
    --raw-output)

oci network route-table update \
    --rt-id "$ROUTE_TABLE_ID" \
    --route-rules '[{"destination": "0.0.0.0/0", "destinationType": "CIDR_BLOCK", "networkEntityId": "'$IG_ID'"}]' \
    --force >/dev/null

# Step 3: Create Security List
echo_step "3/6 Creating Security List..."
SECURITY_LIST_ID=$(oci network security-list create \
    --compartment-id "$TENANCY_ID" \
    --vcn-id "$VCN_ID" \
    --display-name "$SECURITY_LIST_NAME" \
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
    --query 'data.id' \
    --raw-output)
echo_info "Security List created: $SECURITY_LIST_ID"

# Step 4: Create Subnet
echo_step "4/6 Creating Subnet..."
SUBNET_ID=$(oci network subnet create \
    --compartment-id "$TENANCY_ID" \
    --vcn-id "$VCN_ID" \
    --display-name "$SUBNET_NAME" \
    --cidr-block "10.0.1.0/24" \
    --availability-domain "${AVAILABILITY_DOMAINS[0]}" \
    --security-list-ids "[\"$SECURITY_LIST_ID\"]" \
    --query 'data.id' \
    --raw-output)
echo_info "Subnet created: $SUBNET_ID"

# Step 5: Launch Always Free AMD Instance
echo_step "5/6 Launching Always Free AMD Instance..."
echo_info "Configuration: VM.Standard.E2.1.Micro (1/8 OCPU, 1GB RAM)"

# Try each availability domain
for ad in "${AVAILABILITY_DOMAINS[@]}"; do
    echo_info "Trying to launch in $ad..."

    if INSTANCE_ID=$(oci compute instance launch \
        --compartment-id "$TENANCY_ID" \
        --availability-domain "$ad" \
        --shape "VM.Standard.E2.1.Micro" \
        --image-id "$AMD_IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --display-name "$INSTANCE_NAME" \
        --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
        --assign-public-ip true \
        --wait-for-state RUNNING \
        --query 'data.id' \
        --raw-output 2>/dev/null); then
        echo_info "✅ Instance launched successfully in $ad!"
        break
    else
        echo_warn "Failed to launch in $ad, trying next..."
    fi
done

if [ -z "$INSTANCE_ID" ]; then
    echo_error "❌ Could not launch instance in any availability domain"
    exit 1
fi

# Step 6: Get instance details
echo_step "6/6 Getting instance details..."
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
VCN ID: $VCN_ID
Subnet ID: $SUBNET_ID
Security List ID: $SECURITY_LIST_ID
Internet Gateway ID: $IG_ID

Access Information:
- SSH: ssh opc@$PUBLIC_IP
- AIDB will be at: http://$PUBLIC_IP:8080 (after setup)

Setup Commands:
ssh opc@$PUBLIC_IP
# Then run AIDB setup commands
EOF

echo ""
echo "🎉 ALWAYS FREE INSTANCE DEPLOYED!"
echo "=================================="
echo_info "📍 Instance Details:"
echo "   - Shape: VM.Standard.E2.1.Micro (Always Free)"
echo "   - CPU: 1/8 OCPU (~0.125 vCPU)"
echo "   - Memory: 1GB RAM"
echo "   - Storage: Boot volume included"
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
echo ""
echo_info "🗑️ Cleanup (if needed):"
echo "   ./cleanup-oracle-resources.sh"
echo "=================================="