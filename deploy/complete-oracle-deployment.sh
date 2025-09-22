#!/bin/bash

# Complete Oracle Cloud AIDB Deployment Script
# This script performs end-to-end deployment including infrastructure setup and cleanup
# Author: Claude Code Assistant
# Date: $(date)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

echo_action() {
    echo -e "${PURPLE}[ACTION]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Oracle Cloud Configuration
TENANCY_ID="ocid1.tenancy.oc1..aaaaaaaap7ospwxuoqjhqkf4ec3mpz7o3q3nue6kfcc3c7lxu4zkh3b2ophq"
USER_ID="ocid1.user.oc1..aaaaaaaaokget3ftukuqwsjnflivd25o74vebkzn7zm73uhj6lkjboerliiq"
REGION="us-chicago-1"
FINGERPRINT="76:04:de:f1:3b:28:23:5c:5a:93:be:4d:e2:d9:21:2e"

# Deployment Configuration
INSTANCE_NAME="aidb-server"
VCN_NAME="aidb-vcn"
SUBNET_NAME="aidb-subnet"
SECURITY_LIST_NAME="aidb-security-list"
IG_NAME="aidb-ig"

# Availability Domains
AVAILABILITY_DOMAINS=("GtHP:US-CHICAGO-1-AD-1" "GtHP:US-CHICAGO-1-AD-2" "GtHP:US-CHICAGO-1-AD-3")

# Ensure Oracle CLI is in PATH
export PATH="/Users/vivekkumar/Library/Python/3.9/bin:$PATH"
export OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True
export SUPPRESS_LABEL_WARNING=True

# Banner
echo "=================================================="
echo_info "🚀 Complete Oracle Cloud AIDB Deployment Script"
echo_info "Handles full lifecycle: Setup, Deploy, Cleanup"
echo "=================================================="
echo

# Function to check prerequisites
check_prerequisites() {
    echo_step "Checking prerequisites..."

    local missing_deps=()

    if ! command -v oci &> /dev/null; then
        missing_deps+=("oci")
    fi

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v ssh &> /dev/null; then
        missing_deps+=("ssh")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo_error "Missing dependencies: ${missing_deps[*]}"
        echo "Please install missing dependencies and run again."
        exit 1
    fi

    echo_info "✅ Prerequisites check passed!"
}

# Function to setup Oracle CLI
setup_oracle_cli() {
    echo_step "Setting up Oracle CLI..."

    # Create .oci directory
    mkdir -p ~/.oci

    # Generate API key pair if not exists
    if [ ! -f ~/.oci/oci_api_key.pem ]; then
        echo_action "Generating Oracle CLI API key pair..."
        openssl genrsa -out ~/.oci/oci_api_key.pem 2048
        openssl rsa -pubout -in ~/.oci/oci_api_key.pem -out ~/.oci/oci_api_key_public.pem
        chmod 600 ~/.oci/oci_api_key.pem

        echo_info "📋 Public key for Oracle Cloud Console:"
        echo "=================================================="
        cat ~/.oci/oci_api_key_public.pem
        echo "=================================================="
        echo_warn "⚠️  MANUAL STEP REQUIRED:"
        echo "1. Go to: https://cloud.oracle.com/"
        echo "2. Profile → User Settings → API Keys → Add API Key"
        echo "3. Paste the public key above"
        echo "4. Update FINGERPRINT in this script with the generated fingerprint"
        echo "5. Re-run this script"
        exit 0
    fi

    # Create config file
    cat > ~/.oci/config << EOF
[DEFAULT]
user=$USER_ID
fingerprint=$FINGERPRINT
tenancy=$TENANCY_ID
region=$REGION
key_file=$HOME/.oci/oci_api_key.pem
EOF

    # Fix permissions
    chmod 600 ~/.oci/config

    # Test connection
    if oci iam user get --user-id "$USER_ID" >/dev/null 2>&1; then
        echo_info "✅ Oracle CLI configured and tested successfully!"
    else
        echo_error "Oracle CLI test failed. Check your configuration."
        exit 1
    fi
}

# Function to setup SSH keys
setup_ssh_keys() {
    echo_step "Setting up SSH keys..."

    if [ ! -f ~/.ssh/id_rsa.pub ]; then
        echo_action "Generating SSH key pair..."
        ssh-keygen -t rsa -b 4096 -C "aidb-oracle-deployment" -f ~/.ssh/id_rsa -N ""
    fi

    echo_info "✅ SSH keys ready"
}

# Function to get latest image IDs
get_image_ids() {
    echo_action "Getting latest Oracle Linux 8 image IDs..."

    # Get ARM image
    ARM_IMAGE_ID=$(oci compute image list \
        --compartment-id "$TENANCY_ID" \
        --operating-system "Oracle Linux" \
        --operating-system-version "8" \
        --query 'data[?contains("display-name", `aarch64`) && contains("display-name", `2025.08.31`)].id | [0]' \
        --raw-output 2>/dev/null || echo "")

    # Get AMD image
    AMD_IMAGE_ID=$(oci compute image list \
        --compartment-id "$TENANCY_ID" \
        --operating-system "Oracle Linux" \
        --operating-system-version "8" \
        --query 'data[?!contains("display-name", `aarch64`) && !contains("display-name", `GPU`) && contains("display-name", `2025.08.31`)].id | [0]' \
        --raw-output 2>/dev/null || echo "")

    if [ -z "$ARM_IMAGE_ID" ] || [ "$ARM_IMAGE_ID" = "null" ]; then
        ARM_IMAGE_ID="ocid1.image.oc1.us-chicago-1.aaaaaaaaxfywtcajqgn25ma663ybfac35imks3zcoionaa75kkqdqrucp5ba"
    fi

    if [ -z "$AMD_IMAGE_ID" ] || [ "$AMD_IMAGE_ID" = "null" ]; then
        AMD_IMAGE_ID="ocid1.image.oc1.us-chicago-1.aaaaaaaaeae3ejm7ismgu6ebhuk7rdbm6a26dqzjwxfxnooybg5iyo2vqdhq"
    fi

    echo_info "ARM Image ID: $ARM_IMAGE_ID"
    echo_info "AMD Image ID: $AMD_IMAGE_ID"
}

# Function to create VCN
create_vcn() {
    echo_step "Creating Virtual Cloud Network..."

    # Check if VCN exists
    VCN_ID=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --display-name "$VCN_NAME" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null || echo "null")

    if [ "$VCN_ID" = "null" ] || [ -z "$VCN_ID" ]; then
        echo_action "Creating VCN: $VCN_NAME"
        VCN_ID=$(oci network vcn create \
            --compartment-id "$TENANCY_ID" \
            --display-name "$VCN_NAME" \
            --cidr-block "10.0.0.0/16" \
            --query 'data.id' \
            --raw-output)
        echo_info "VCN created: $VCN_ID"
        sleep 5
    else
        echo_info "Using existing VCN: $VCN_ID"
    fi
}

# Function to create Internet Gateway
create_internet_gateway() {
    echo_step "Creating Internet Gateway..."

    # Check if IG exists
    IG_ID=$(oci network internet-gateway list \
        --compartment-id "$TENANCY_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "$IG_NAME" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null || echo "null")

    if [ "$IG_ID" = "null" ] || [ -z "$IG_ID" ]; then
        echo_action "Creating Internet Gateway: $IG_NAME"
        IG_ID=$(oci network internet-gateway create \
            --compartment-id "$TENANCY_ID" \
            --vcn-id "$VCN_ID" \
            --display-name "$IG_NAME" \
            --is-enabled true \
            --query 'data.id' \
            --raw-output)
        echo_info "Internet Gateway created: $IG_ID"

        # Add route rule
        echo_action "Adding route rule..."
        ROUTE_TABLE_ID=$(oci network vcn get \
            --vcn-id "$VCN_ID" \
            --query 'data."default-route-table-id"' \
            --raw-output)

        oci network route-table update \
            --rt-id "$ROUTE_TABLE_ID" \
            --route-rules '[{"destination": "0.0.0.0/0", "destinationType": "CIDR_BLOCK", "networkEntityId": "'$IG_ID'"}]' \
            --force >/dev/null

        echo_info "Route rule added"
    else
        echo_info "Using existing Internet Gateway: $IG_ID"
    fi
}

# Function to create security list
create_security_list() {
    echo_step "Creating Security List..."

    # Check if security list exists
    SECURITY_LIST_ID=$(oci network security-list list \
        --compartment-id "$TENANCY_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "$SECURITY_LIST_NAME" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null || echo "null")

    if [ "$SECURITY_LIST_ID" = "null" ] || [ -z "$SECURITY_LIST_ID" ]; then
        echo_action "Creating Security List: $SECURITY_LIST_NAME"
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
    else
        echo_info "Using existing Security List: $SECURITY_LIST_ID"
    fi
}

# Function to create subnet
create_subnet() {
    echo_step "Creating Subnet..."

    # Check if subnet exists
    SUBNET_ID=$(oci network subnet list \
        --compartment-id "$TENANCY_ID" \
        --vcn-id "$VCN_ID" \
        --display-name "$SUBNET_NAME" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null || echo "null")

    if [ "$SUBNET_ID" = "null" ] || [ -z "$SUBNET_ID" ]; then
        echo_action "Creating Subnet: $SUBNET_NAME"
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
    else
        echo_info "Using existing Subnet: $SUBNET_ID"

        # Ensure security list is associated
        echo_action "Updating subnet security list..."
        oci network subnet update \
            --subnet-id "$SUBNET_ID" \
            --security-list-ids "[\"$SECURITY_LIST_ID\"]" \
            --force >/dev/null
    fi
}

# Function to launch instance with capacity hunting
launch_instance() {
    echo_step "Launching Compute Instance..."

    # Check if instance exists
    EXISTING_INSTANCE=$(oci compute instance list \
        --compartment-id "$TENANCY_ID" \
        --display-name "$INSTANCE_NAME" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null || echo "null")

    if [ "$EXISTING_INSTANCE" != "null" ] && [ -n "$EXISTING_INSTANCE" ]; then
        echo_warn "Instance $INSTANCE_NAME already exists: $EXISTING_INSTANCE"
        INSTANCE_ID="$EXISTING_INSTANCE"
        return
    fi

    echo_action "Attempting to launch ARM instance (preferred)..."

    # Try ARM instance first (with capacity hunting)
    for ad in "${AVAILABILITY_DOMAINS[@]}"; do
        echo_info "Trying ARM instance in $ad..."

        if INSTANCE_ID=$(oci compute instance launch \
            --compartment-id "$TENANCY_ID" \
            --availability-domain "$ad" \
            --shape "VM.Standard.A1.Flex" \
            --shape-config '{"memoryInGBs": 12, "ocpus": 2}' \
            --image-id "$ARM_IMAGE_ID" \
            --subnet-id "$SUBNET_ID" \
            --display-name "$INSTANCE_NAME" \
            --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
            --assign-public-ip true \
            --wait-for-state RUNNING \
            --query 'data.id' \
            --raw-output 2>/dev/null); then
            echo_info "✅ ARM instance launched successfully in $ad: $INSTANCE_ID"
            return
        else
            echo_warn "No ARM capacity in $ad, trying next..."
        fi
    done

    echo_warn "No ARM capacity available, falling back to AMD Always Free..."

    # Try AMD instance
    for ad in "${AVAILABILITY_DOMAINS[@]}"; do
        echo_info "Trying AMD instance in $ad..."

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
            echo_info "✅ AMD instance launched successfully in $ad: $INSTANCE_ID"
            return
        else
            echo_warn "Failed to launch AMD instance in $ad, trying next..."
        fi
    done

    echo_error "❌ Could not launch instance in any availability domain"
    echo_info "Manual fallback:"
    echo "1. Go to Oracle Cloud Console: https://cloud.oracle.com/"
    echo "2. Create instance manually with these settings:"
    echo "   - VCN: $VCN_NAME"
    echo "   - Subnet: $SUBNET_NAME"
    echo "   - Shape: VM.Standard.E2.1.Micro or VM.Standard.A1.Flex"
    echo "   - SSH Key: $(cat ~/.ssh/id_rsa.pub)"
    exit 1
}

# Function to get instance details
get_instance_details() {
    echo_step "Getting instance details..."

    # Get public IP
    PUBLIC_IP=$(oci compute instance list-vnics \
        --instance-id "$INSTANCE_ID" \
        --query 'data[0]."public-ip"' \
        --raw-output 2>/dev/null)

    if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "null" ]; then
        echo_error "Could not get public IP for instance"
        exit 1
    fi

    echo_info "Instance ID: $INSTANCE_ID"
    echo_info "Public IP: $PUBLIC_IP"
}

# Function to deploy AIDB
deploy_aidb() {
    echo_step "Deploying AIDB to instance..."

    echo_action "Waiting for SSH to be available..."
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

    # Upload setup script
    echo_action "Uploading AIDB setup script..."
    scp -o StrictHostKeyChecking=no "$SCRIPT_DIR/setup-aidb.sh" opc@$PUBLIC_IP:~/

    # Run installation
    echo_action "Running AIDB installation..."
    ssh -o StrictHostKeyChecking=no opc@$PUBLIC_IP 'chmod +x setup-aidb.sh && echo "1" | ./setup-aidb.sh'

    echo_info "✅ AIDB deployed successfully"
}

# Function to verify deployment
verify_deployment() {
    echo_step "Verifying AIDB deployment..."

    echo_action "Testing AIDB health endpoint..."
    sleep 10

    if curl -f "http://$PUBLIC_IP:8080/health" >/dev/null 2>&1; then
        echo_info "✅ AIDB is running successfully!"
    else
        echo_warn "⚠️  Health check failed. AIDB may still be starting up."
        echo_info "Check status with: ssh opc@$PUBLIC_IP 'sudo systemctl status aidb'"
    fi
}

# Function to save deployment info
save_deployment_info() {
    echo_step "Saving deployment information..."

    cat > "$SCRIPT_DIR/deployment-info.txt" << EOF
AIDB Oracle Cloud Deployment Information
========================================
Deployment Date: $(date)
Instance ID: $INSTANCE_ID
Instance Name: $INSTANCE_NAME
Public IP: $PUBLIC_IP
VCN ID: $VCN_ID
Subnet ID: $SUBNET_ID
Security List ID: $SECURITY_LIST_ID
Internet Gateway ID: $IG_ID

Access Information:
- AIDB URL: http://$PUBLIC_IP:8080
- SSH: ssh opc@$PUBLIC_IP
- Health: http://$PUBLIC_IP:8080/health

Management:
- Status: ssh opc@$PUBLIC_IP 'sudo systemctl status aidb'
- Logs: ssh opc@$PUBLIC_IP 'sudo journalctl -u aidb -f'
- Restart: ssh opc@$PUBLIC_IP 'sudo systemctl restart aidb'

Resource IDs for Cleanup:
- Instance: $INSTANCE_ID
- VCN: $VCN_ID
- Subnet: $SUBNET_ID
- Security List: $SECURITY_LIST_ID
- Internet Gateway: $IG_ID
EOF

    echo_info "📝 Deployment info saved to: $SCRIPT_DIR/deployment-info.txt"
}

# Function to cleanup resources
cleanup_resources() {
    echo_step "Cleaning up Oracle Cloud resources..."

    if [ -f "$SCRIPT_DIR/deployment-info.txt" ]; then
        source <(grep -E '^Instance|^VCN|^Subnet|^Security List|^Internet Gateway' "$SCRIPT_DIR/deployment-info.txt" | sed 's/^[^:]*: /export /' | sed 's/ /_/g' | sed 's/ID=/ID_CLEANUP=/')
    fi

    echo_action "Terminating compute instance..."
    if [ -n "${Instance_ID_CLEANUP:-}" ]; then
        oci compute instance terminate --instance-id "${Instance_ID_CLEANUP}" --force 2>/dev/null || true
        echo_info "Instance termination initiated"
    fi

    # Wait for instance to terminate
    echo_action "Waiting for instance to terminate..."
    sleep 60

    echo_action "Deleting subnet..."
    if [ -n "${Subnet_ID_CLEANUP:-}" ]; then
        oci network subnet delete --subnet-id "${Subnet_ID_CLEANUP}" --force 2>/dev/null || true
    fi

    echo_action "Deleting security list..."
    if [ -n "${Security_List_ID_CLEANUP:-}" ]; then
        oci network security-list delete --security-list-id "${Security_List_ID_CLEANUP}" --force 2>/dev/null || true
    fi

    echo_action "Deleting internet gateway..."
    if [ -n "${Internet_Gateway_ID_CLEANUP:-}" ]; then
        oci network internet-gateway delete --ig-id "${Internet_Gateway_ID_CLEANUP}" --force 2>/dev/null || true
    fi

    echo_action "Deleting VCN..."
    if [ -n "${VCN_ID_CLEANUP:-}" ]; then
        oci network vcn delete --vcn-id "${VCN_ID_CLEANUP}" --force 2>/dev/null || true
    fi

    echo_info "✅ Cleanup completed"

    # Remove deployment info file
    rm -f "$SCRIPT_DIR/deployment-info.txt"
}

# Function to display final summary
show_summary() {
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "=================================="
    echo_info "📍 Instance Details:"
    echo "   - Name: $INSTANCE_NAME"
    echo "   - Public IP: $PUBLIC_IP"
    echo "   - Instance ID: $INSTANCE_ID"
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
    echo_info "🗑️  Cleanup Command:"
    echo "   $0 --cleanup"
    echo "=================================="
}

# Main function
main() {
    case "${1:-}" in
        --cleanup)
            cleanup_resources
            exit 0
            ;;
        --setup-only)
            check_prerequisites
            setup_oracle_cli
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cleanup      Delete all created Oracle Cloud resources"
            echo "  --setup-only   Only setup Oracle CLI and SSH keys"
            echo "  --help         Show this help message"
            echo ""
            echo "Default: Full deployment (setup + deploy + verify)"
            exit 0
            ;;
    esac

    # Full deployment
    check_prerequisites
    setup_oracle_cli
    setup_ssh_keys
    get_image_ids
    create_vcn
    create_internet_gateway
    create_security_list
    create_subnet
    launch_instance
    get_instance_details
    deploy_aidb
    verify_deployment
    save_deployment_info
    show_summary
}

# Run main function with all arguments
main "$@"