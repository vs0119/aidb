#!/bin/bash

# Oracle Cloud Resources Cleanup Script
# This script deletes ONLY Oracle Cloud infrastructure resources
# Does NOT touch local files, SSH keys, or Oracle CLI configuration

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
REGION="us-chicago-1"

# Resource names to search for
INSTANCE_NAME="aidb-server"
VCN_NAME="aidb-vcn"
SUBNET_NAME="aidb-subnet"
SECURITY_LIST_NAME="aidb-security-list"
IG_NAME="aidb-ig"

# Ensure Oracle CLI is in PATH
export PATH="/Users/vivekkumar/Library/Python/3.9/bin:$PATH"
export OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING=True
export SUPPRESS_LABEL_WARNING=True

# Banner
echo "=================================================="
echo_info "🗑️  Oracle Cloud Resources Cleanup Script"
echo_warn "⚠️  This will DELETE Oracle Cloud resources ONLY"
echo_info "✅ Local files and configurations are SAFE"
echo "=================================================="
echo

# Function to check if Oracle CLI is working
check_oracle_cli() {
    echo_step "Checking Oracle CLI..."

    if ! command -v oci &> /dev/null; then
        echo_error "Oracle CLI not found in PATH"
        exit 1
    fi

    if ! oci iam user list --compartment-id "$TENANCY_ID" >/dev/null 2>&1; then
        echo_error "Oracle CLI not configured or credentials invalid"
        exit 1
    fi

    echo_info "✅ Oracle CLI is working"
}

# Function to find and terminate instances
cleanup_instances() {
    echo_step "Finding and terminating compute instances..."

    # Find all instances with our name
    INSTANCE_IDS=$(oci compute instance list \
        --compartment-id "$TENANCY_ID" \
        --display-name "$INSTANCE_NAME" \
        --query 'data[*].id' \
        --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

    if [ -z "$INSTANCE_IDS" ]; then
        echo_info "No instances named '$INSTANCE_NAME' found"
    else
        echo "$INSTANCE_IDS" | while read -r instance_id; do
            if [ -n "$instance_id" ] && [ "$instance_id" != "null" ]; then
                echo_info "Terminating instance: $instance_id"
                oci compute instance terminate \
                    --instance-id "$instance_id" \
                    --force 2>/dev/null || echo_warn "Failed to terminate instance $instance_id"
            fi
        done
        echo_info "Instance termination initiated (takes 1-2 minutes)"
        echo_info "Waiting 60 seconds for instances to terminate..."
        sleep 60
    fi
}

# Function to delete subnets
cleanup_subnets() {
    echo_step "Finding and deleting subnets..."

    # Find all VCNs first
    VCN_IDS=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].id' \
        --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

    if [ -z "$VCN_IDS" ]; then
        echo_info "No VCNs found"
        return
    fi

    echo "$VCN_IDS" | while read -r vcn_id; do
        if [ -n "$vcn_id" ] && [ "$vcn_id" != "null" ]; then
            # Find subnets in this VCN
            SUBNET_IDS=$(oci network subnet list \
                --compartment-id "$TENANCY_ID" \
                --vcn-id "$vcn_id" \
                --query 'data[*].id' \
                --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

            if [ -n "$SUBNET_IDS" ]; then
                echo "$SUBNET_IDS" | while read -r subnet_id; do
                    if [ -n "$subnet_id" ] && [ "$subnet_id" != "null" ]; then
                        echo_info "Deleting subnet: $subnet_id"
                        oci network subnet delete \
                            --subnet-id "$subnet_id" \
                            --force 2>/dev/null || echo_warn "Failed to delete subnet $subnet_id"
                    fi
                done
            fi
        fi
    done
}

# Function to delete security lists
cleanup_security_lists() {
    echo_step "Finding and deleting custom security lists..."

    # Find all VCNs first
    VCN_IDS=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].id' \
        --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

    if [ -z "$VCN_IDS" ]; then
        echo_info "No VCNs found"
        return
    fi

    echo "$VCN_IDS" | while read -r vcn_id; do
        if [ -n "$vcn_id" ] && [ "$vcn_id" != "null" ]; then
            # Find custom security lists (exclude default)
            SECURITY_LIST_IDS=$(oci network security-list list \
                --compartment-id "$TENANCY_ID" \
                --vcn-id "$vcn_id" \
                --query 'data[?!"display-name" || !contains("display-name", `Default`)].id' \
                --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

            if [ -n "$SECURITY_LIST_IDS" ]; then
                echo "$SECURITY_LIST_IDS" | while read -r security_list_id; do
                    if [ -n "$security_list_id" ] && [ "$security_list_id" != "null" ]; then
                        echo_info "Deleting security list: $security_list_id"
                        oci network security-list delete \
                            --security-list-id "$security_list_id" \
                            --force 2>/dev/null || echo_warn "Failed to delete security list $security_list_id"
                    fi
                done
            fi
        fi
    done
}

# Function to delete internet gateways
cleanup_internet_gateways() {
    echo_step "Finding and deleting internet gateways..."

    # Find all VCNs first
    VCN_IDS=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].id' \
        --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

    if [ -z "$VCN_IDS" ]; then
        echo_info "No VCNs found"
        return
    fi

    echo "$VCN_IDS" | while read -r vcn_id; do
        if [ -n "$vcn_id" ] && [ "$vcn_id" != "null" ]; then
            # Find internet gateways in this VCN
            IG_IDS=$(oci network internet-gateway list \
                --compartment-id "$TENANCY_ID" \
                --vcn-id "$vcn_id" \
                --query 'data[*].id' \
                --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

            if [ -n "$IG_IDS" ]; then
                echo "$IG_IDS" | while read -r ig_id; do
                    if [ -n "$ig_id" ] && [ "$ig_id" != "null" ]; then
                        echo_info "Deleting internet gateway: $ig_id"
                        oci network internet-gateway delete \
                            --ig-id "$ig_id" \
                            --force 2>/dev/null || echo_warn "Failed to delete internet gateway $ig_id"
                    fi
                done
            fi
        fi
    done
}

# Function to delete VCNs
cleanup_vcns() {
    echo_step "Finding and deleting VCNs..."

    # Find all VCNs
    VCN_IDS=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].id' \
        --raw-output 2>/dev/null | tr ',' '\n' | grep -v '^$' || echo "")

    if [ -z "$VCN_IDS" ]; then
        echo_info "No VCNs found"
        return
    fi

    echo "$VCN_IDS" | while read -r vcn_id; do
        if [ -n "$vcn_id" ] && [ "$vcn_id" != "null" ]; then
            echo_info "Deleting VCN: $vcn_id"
            oci network vcn delete \
                --vcn-id "$vcn_id" \
                --force 2>/dev/null || echo_warn "Failed to delete VCN $vcn_id"
        fi
    done
}

# Function to show what will be deleted
show_resources() {
    echo_step "Scanning for Oracle Cloud resources to delete..."
    echo ""

    # Show instances
    echo_info "🖥️  Compute Instances:"
    INSTANCES=$(oci compute instance list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].{"Name":"display-name","State":"lifecycle-state","ID":id}' \
        --output table 2>/dev/null || echo "No instances found")
    echo "$INSTANCES"
    echo ""

    # Show VCNs
    echo_info "🌐 Virtual Cloud Networks:"
    VCNS=$(oci network vcn list \
        --compartment-id "$TENANCY_ID" \
        --query 'data[*].{"Name":"display-name","ID":id}' \
        --output table 2>/dev/null || echo "No VCNs found")
    echo "$VCNS"
    echo ""
}

# Function to confirm deletion
confirm_deletion() {
    echo_warn "⚠️  WARNING: This will permanently delete all Oracle Cloud resources shown above!"
    echo_info "Local files, SSH keys, and Oracle CLI config will NOT be touched."
    echo ""
    read -p "Are you sure you want to proceed? (type 'yes' to confirm): " confirmation

    if [ "$confirmation" != "yes" ]; then
        echo_info "Cleanup cancelled."
        exit 0
    fi
}

# Main cleanup function
cleanup_all_resources() {
    echo_step "Starting Oracle Cloud resource cleanup..."
    echo ""

    cleanup_instances
    echo ""

    cleanup_subnets
    echo ""

    cleanup_security_lists
    echo ""

    cleanup_internet_gateways
    echo ""

    cleanup_vcns
    echo ""

    echo_info "✅ Oracle Cloud resource cleanup completed!"
}

# Main function
main() {
    case "${1:-}" in
        --list-only)
            check_oracle_cli
            show_resources
            exit 0
            ;;
        --force)
            check_oracle_cli
            cleanup_all_resources
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list-only    Show what resources would be deleted (no deletion)"
            echo "  --force        Delete without confirmation prompt"
            echo "  --help         Show this help message"
            echo ""
            echo "Default: Show resources and ask for confirmation before deletion"
            exit 0
            ;;
    esac

    # Default: interactive mode
    check_oracle_cli
    show_resources
    confirm_deletion
    cleanup_all_resources
}

# Run main function with all arguments
main "$@"