#!/bin/bash

# Oracle Cloud ARM Capacity Hunter
# This script automatically retries creating ARM instances when capacity becomes available

echo "🔍 Oracle ARM Capacity Hunter"
echo "This script will keep trying to create your ARM instance until capacity is available"
echo ""

# Configuration - Update these with your details
COMPARTMENT_ID="your-compartment-ocid"
AVAILABILITY_DOMAINS=("AD-1" "AD-2" "AD-3")
SUBNET_ID="your-subnet-ocid"
IMAGE_ID="ocid1.image.oc1.iad.aaaaaaaag2uyozo7266bmg26j5rrivi62spram6sps2u3ikbt3c3yrmqmeaa"
SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"

echo "⚙️  Configuration:"
echo "   - Shape: VM.Standard.A1.Flex (ARM)"
echo "   - Memory: 6GB"
echo "   - OCPU: 1"
echo "   - Retry interval: 30 seconds"
echo ""

# Check if oci CLI is configured
if ! command -v oci &> /dev/null; then
    echo "❌ Oracle CLI not found. Please install and configure it first."
    exit 1
fi

# Function to try creating instance
try_create_instance() {
    local ad=$1
    echo "🔄 Trying to create instance in $ad..."

    # Try to create the instance
    oci compute instance launch \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$ad" \
        --shape "VM.Standard.A1.Flex" \
        --shape-config '{"memoryInGBs": 6, "ocpus": 1}' \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --display-name "aidb-server-arm" \
        --ssh-authorized-keys-file "$SSH_KEY_PATH" \
        --assign-public-ip true \
        2>/dev/null

    return $?
}

# Main hunting loop
attempt=1
while true; do
    echo "🎯 Attempt #$attempt - $(date '+%Y-%m-%d %H:%M:%S')"

    # Try each availability domain
    for ad in "${AVAILABILITY_DOMAINS[@]}"; do
        if try_create_instance "$ad"; then
            echo "🎉 SUCCESS! ARM instance created in $ad"
            echo "🔍 Getting instance details..."

            # Get the instance details
            oci compute instance list \
                --compartment-id "$COMPARTMENT_ID" \
                --display-name "aidb-server-arm" \
                --lifecycle-state RUNNING \
                --query 'data[0].{"Name":"display-name","State":"lifecycle-state","IP":"public-ip"}' \
                --output table

            echo "✅ Instance creation completed!"
            echo "📋 Next steps:"
            echo "   1. Wait for instance to fully boot (2-3 minutes)"
            echo "   2. SSH to your instance: ssh opc@<public-ip>"
            echo "   3. Run the AIDB setup script"
            exit 0
        else
            echo "   ❌ No capacity in $ad"
        fi
    done

    echo "⏱️  No capacity available. Retrying in 30 seconds..."
    echo "   💡 Tip: ARM instances are popular. Best success rates are early morning or late evening."
    echo ""

    sleep 30
    ((attempt++))

    # Optional: Stop after certain number of attempts
    if [ $attempt -gt 120 ]; then  # Stop after 1 hour
        echo "⏰ Reached maximum attempts. Consider using AMD instance instead."
        break
    fi
done

echo ""
echo "🔄 Alternative: Use AMD Always Free Instance"
echo "   Shape: VM.Standard.E2.1.Micro (1/8 OCPU, 1GB RAM)"
echo "   ✅ Always available, still free forever"