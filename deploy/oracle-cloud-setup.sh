#!/bin/bash

# Oracle Cloud Infrastructure Setup Script for AIDB
# This script sets up AIDB on Oracle Cloud Always Free tier

set -e

# Configuration
INSTANCE_NAME="aidb-server"
COMPARTMENT_ID=""  # Set your compartment OCID
IMAGE_ID="ocid1.image.oc1.iad.aaaaaaaag2uyozo7266bmg26j5rrivi62spram6sps2u3ikbt3c3yrmqmeaa"  # Oracle Linux 8 AMD64
SHAPE="VM.Standard.E2.1.Micro"  # Always Free shape
SSH_KEY_PATH="~/.ssh/id_rsa.pub"
AVAILABILITY_DOMAIN=""  # Set your AD (e.g., "Uocm:US-ASHBURN-AD-1")

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

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."

    if ! command -v oci &> /dev/null; then
        echo_error "Oracle CLI not found. Please install it first:"
        echo "curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh | bash"
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

    echo_info "Prerequisites check passed!"
}

# Validate configuration
validate_config() {
    echo_info "Validating configuration..."

    if [ -z "$COMPARTMENT_ID" ]; then
        echo_error "COMPARTMENT_ID not set. Get it from Oracle Cloud Console."
        exit 1
    fi

    if [ -z "$AVAILABILITY_DOMAIN" ]; then
        echo_error "AVAILABILITY_DOMAIN not set. Get it from Oracle Cloud Console."
        exit 1
    fi

    echo_info "Configuration validation passed!"
}

# Create VCN and subnet if they don't exist
setup_network() {
    echo_info "Setting up network infrastructure..."

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
}

# Create security list with required ports
setup_security() {
    echo_info "Setting up security rules..."

    # Create security list
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
        --query 'data.id' --raw-output 2>/dev/null || true)

    echo_info "Security list configured"
}

# Launch compute instance
launch_instance() {
    echo_info "Launching compute instance..."

    # Check if instance already exists
    EXISTING_INSTANCE=$(oci compute instance list --compartment-id "$COMPARTMENT_ID" --display-name "$INSTANCE_NAME" --query 'data[0].id' --raw-output 2>/dev/null || echo "null")

    if [ "$EXISTING_INSTANCE" != "null" ]; then
        echo_warn "Instance $INSTANCE_NAME already exists: $EXISTING_INSTANCE"
        return
    fi

    # Launch instance
    INSTANCE_ID=$(oci compute instance launch \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --shape "$SHAPE" \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --display-name "$INSTANCE_NAME" \
        --ssh-authorized-keys-file "$SSH_KEY_PATH" \
        --wait-for-state RUNNING \
        --query 'data.id' --raw-output)

    echo_info "Instance launched successfully: $INSTANCE_ID"

    # Get public IP
    PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output)
    echo_info "Public IP: $PUBLIC_IP"

    echo_info "Waiting for instance to be ready for SSH..."
    sleep 60

    echo_info "Instance setup complete!"
    echo_info "Connect via SSH: ssh opc@$PUBLIC_IP"
}

# Deploy AIDB to the instance
deploy_aidb() {
    echo_info "Deploying AIDB..."

    # Get instance public IP
    INSTANCE_ID=$(oci compute instance list --compartment-id "$COMPARTMENT_ID" --display-name "$INSTANCE_NAME" --query 'data[0].id' --raw-output)
    PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output)

    if [ "$PUBLIC_IP" = "null" ]; then
        echo_error "Could not get public IP for instance"
        exit 1
    fi

    echo_info "Deploying to instance at $PUBLIC_IP"

    # Upload deployment script
    scp -o StrictHostKeyChecking=no deploy/setup-aidb.sh opc@$PUBLIC_IP:~/

    # Run deployment
    ssh -o StrictHostKeyChecking=no opc@$PUBLIC_IP 'chmod +x setup-aidb.sh && ./setup-aidb.sh'

    echo_info "AIDB deployed successfully!"
    echo_info "Access AIDB at: http://$PUBLIC_IP:8080"
}

# Main execution
main() {
    echo_info "Starting Oracle Cloud AIDB deployment..."

    check_prerequisites
    validate_config
    setup_network
    setup_security
    launch_instance
    deploy_aidb

    echo_info "Deployment complete!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --compartment-id)
            COMPARTMENT_ID="$2"
            shift 2
            ;;
        --availability-domain)
            AVAILABILITY_DOMAIN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --compartment-id OCID --availability-domain AD"
            echo "Example: $0 --compartment-id ocid1.compartment.oc1..xxx --availability-domain Uocm:US-ASHBURN-AD-1"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

main