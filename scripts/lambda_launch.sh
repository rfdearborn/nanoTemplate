#!/bin/bash
set -e

# =============================================================================
# Lambda Labs Training Launcher
# =============================================================================
# Usage:
#   ./scripts/lambda_launch.sh launch [config_file]    - Spin up instance and start training
#   ./scripts/lambda_launch.sh setup [config_file]     - Resume setup/restart training on active instance
#   ./scripts/lambda_launch.sh status                  - Check running instances
#   ./scripts/lambda_launch.sh ssh                     - SSH into the running instance
#   ./scripts/lambda_launch.sh logs                    - Tail training logs
#   ./scripts/lambda_launch.sh stop                    - Terminate the instance
#
# Note: 'launch' will automatically retry every 10s if no instances are available.
#       'setup' is useful for updating code or resuming after a connection failure.
#
# Environment variables (or set below):
#   LAMBDA_API_KEY     - Your Lambda Labs API key
#   HF_TOKEN           - HuggingFace token
#   WANDB_API_KEY      - Weights & Biases API key
# =============================================================================

# Configuration - edit these or set via environment
LAMBDA_API_KEY="${LAMBDA_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Lambda settings
INSTANCE_TYPE="gpu_1x_gh200"
REGION="us-east-3"
SSH_KEY_NAME="Rob's MacBook Pro"
SSH_KEY_PATH="$HOME/.ssh/lambda_labs.pem"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
INSTANCE_FILE="$SCRIPT_DIR/.lambda_instance_id"

# Remote paths
REMOTE_USER="ubuntu"
REMOTE_DIR="~/nanoTemplate"

# API base
API_BASE="https://cloud.lambdalabs.com/api/v1"

# =============================================================================
# Helper functions
# =============================================================================

check_api_key() {
    if [ -z "$LAMBDA_API_KEY" ]; then
        echo "Error: LAMBDA_API_KEY not set"
        echo "Get your API key from: https://cloud.lambdalabs.com/api-keys"
        echo "Then: export LAMBDA_API_KEY='your-key-here'"
        exit 1
    fi
}

api_call() {
    local method=$1
    local endpoint=$2
    local data=$3

    if [ -n "$data" ]; then
        curl -s -X "$method" "$API_BASE$endpoint" \
            -H "Authorization: Bearer $LAMBDA_API_KEY" \
            -H "Content-Type: application/json" \
            -d "$data"
    else
        curl -s -X "$method" "$API_BASE$endpoint" \
            -H "Authorization: Bearer $LAMBDA_API_KEY"
    fi
}

get_instance_ip() {
    local instance_id=$1
    api_call GET "/instances/$instance_id" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('ip',''))"
}

get_instance_status() {
    local instance_id=$1
    api_call GET "/instances/$instance_id" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('status',''))"
}

wait_for_instance() {
    local instance_id=$1
    echo "Waiting for instance to be ready..."

    for i in {1..60}; do
        local status=$(get_instance_status "$instance_id")
        local ip=$(get_instance_ip "$instance_id")

        if [ "$status" = "active" ] && [ -n "$ip" ]; then
            echo "Instance is active at $ip"

            # Wait for SSH to be ready
            echo "Waiting for SSH..."
            for j in {1..30}; do
                if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_USER@$ip" "echo ready" 2>/dev/null; then
                    echo "SSH is ready!"
                    return 0
                fi
                sleep 5
            done
            echo "SSH timeout"
            return 1
        fi

        echo "  Status: $status (attempt $i/60)"
        sleep 10
    done

    echo "Timeout waiting for instance"
    return 1
}

run_setup() {
    local ip=$1
    local config_file="${2:-config/train_gpt2_from_scratch.py}"

    echo "Setting up instance at $ip..."

    # Copy repo
    echo "Syncing repository..."
    rsync -avz --exclude='.git' --exclude='venv' --exclude='wandb' --exclude='out*' --exclude='__pycache__' \
        -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no" \
        "$REPO_DIR/" "$REMOTE_USER@$ip:$REMOTE_DIR/"

    # Setup environment and run training
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$ip" bash -s << REMOTE_SCRIPT
set -e
cd $REMOTE_DIR

# Create and activate virtual environment
echo "Setting up virtual environment..."
rm -rf venv
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q tiktoken datasets wandb deepeval requests huggingface_hub zstandard

# Verify CUDA
echo "Verifying CUDA availability..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CRITICAL: CUDA NOT AVAILABLE!'"

# Login to services
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)"
fi

if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into wandb..."
    python3 -c "import wandb; wandb.login(key='$WANDB_API_KEY')"
fi

# Save source code snapshot for reproducibility
echo "Saving source code snapshot..."
mkdir -p run_snapshots
SNAPSHOT_DIR="run_snapshots/\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$SNAPSHOT_DIR"
cp model.py train.py utils.py configurator.py "\$SNAPSHOT_DIR/"
cp $config_file "\$SNAPSHOT_DIR/"
echo "Config: $config_file" > "\$SNAPSHOT_DIR/run_info.txt"
echo "Date: \$(date)" >> "\$SNAPSHOT_DIR/run_info.txt"
echo "Snapshot saved to \$SNAPSHOT_DIR"

# Log as wandb artifact
python3 << ARTIFACT_SCRIPT
import wandb
import os
import glob

snapshot_dir = "\$SNAPSHOT_DIR"
config_name = os.path.basename("$config_file").replace(".py", "")

run = wandb.init(project="nanoTemplate", job_type="source-snapshot", name=f"source-{config_name}")

artifact = wandb.Artifact(
    name=f"source-{config_name}",
    type="source-code",
    description="Source code snapshot: $config_file"
)

for f in glob.glob(os.path.join(snapshot_dir, "*")):
    artifact.add_file(f)

run.log_artifact(artifact)
run.finish()
print("Source artifact logged to wandb")
ARTIFACT_SCRIPT

# Start training in tmux (persists after SSH disconnect)
echo "Starting training in tmux session 'train'..."
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "cd $REMOTE_DIR && source venv/bin/activate && python3 train.py $config_file --compile=True 2>&1 | tee train.log"

echo ""
echo "=========================================="
echo "Training started in tmux session 'train'"
echo "=========================================="
echo ""
echo "You can now disconnect. The training will continue."
echo ""
echo "Useful commands:"
echo "  ./scripts/lambda_launch.sh ssh    - Connect to instance"
echo "  ./scripts/lambda_launch.sh logs   - Tail training logs"
echo "  ./scripts/lambda_launch.sh stop   - Terminate instance"
echo ""
echo "Once connected via SSH:"
echo "  tmux attach -t train              - Attach to training session"
echo "  Ctrl+B then D                     - Detach from tmux"
echo ""
REMOTE_SCRIPT
}

# =============================================================================
# Commands
# =============================================================================

cmd_launch() {
    local config_file="${1:-config/train_gpt2_from_scratch.py}"

    check_api_key

    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN not set - HuggingFace login will be skipped"
    fi
    if [ -z "$WANDB_API_KEY" ]; then
        echo "Warning: WANDB_API_KEY not set - wandb login will be skipped"
    fi

    # Check for existing instance
    if [ -f "$INSTANCE_FILE" ]; then
        local existing_id=$(cat "$INSTANCE_FILE")
        local existing_status=$(get_instance_status "$existing_id" 2>/dev/null)
        if [ "$existing_status" = "active" ]; then
            echo "Error: Instance $existing_id is already running"
            echo "Use './scripts/lambda_launch.sh stop' first, or './scripts/lambda_launch.sh ssh' to connect"
            exit 1
        fi
    fi

    # List available instance types
    echo "Checking availability for $INSTANCE_TYPE..."
    
    while true; do
        local available=$(api_call GET "/instance-types" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
for name, info in data.items():
    if name == '$INSTANCE_TYPE':
        regions = [r['name'] for r in info.get('regions_with_capacity_available', [])]
        print(','.join(regions) if regions else '')
")

        if [ -n "$available" ]; then
            # Use first available region
            REGION=$(echo "$available" | cut -d',' -f1)
            echo "Found available instance in region: $REGION"
            break
        fi

        echo "No $INSTANCE_TYPE instances available. Retrying in 10 seconds... (Ctrl+C to cancel)"
        sleep 10
    done

    # Launch instance
    echo "Launching $INSTANCE_TYPE instance..."
    local launch_response=$(api_call POST "/instance-operations/launch" "{
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"region_name\": \"$REGION\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"quantity\": 1
    }")

    local instance_id=$(echo "$launch_response" | python3 -c "import sys,json; d=json.load(sys.stdin); ids=d.get('data',{}).get('instance_ids',[]); print(ids[0] if ids else '')")

    if [ -z "$instance_id" ]; then
        echo "Failed to launch instance:"
        echo "$launch_response" | python3 -m json.tool
        exit 1
    fi

    echo "Instance ID: $instance_id"
    echo "$instance_id" > "$INSTANCE_FILE"

    # Wait for instance
    if ! wait_for_instance "$instance_id"; then
        echo "Failed to start instance"
        exit 1
    fi

    local ip=$(get_instance_ip "$instance_id")

    # Setup instance
    run_setup "$ip" "$config_file"

    echo ""
    echo "Instance IP: $ip"
    echo "Instance ID: $instance_id"
}

cmd_status() {
    check_api_key

    echo "Running instances:"
    api_call GET "/instances" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
if not data:
    print('  No running instances')
else:
    for inst in data:
        print(f\"  {inst['id']}: {inst['instance_type']['name']} - {inst['status']} - {inst.get('ip', 'no ip')}\")
"

    if [ -f "$INSTANCE_FILE" ]; then
        echo ""
        echo "Tracked instance: $(cat "$INSTANCE_FILE")"
    fi
}

cmd_ssh() {
    check_api_key

    if [ ! -f "$INSTANCE_FILE" ]; then
        echo "No tracked instance. Use './scripts/lambda_launch.sh status' to see running instances."
        exit 1
    fi

    local instance_id=$(cat "$INSTANCE_FILE")
    local ip=$(get_instance_ip "$instance_id")

    if [ -z "$ip" ]; then
        echo "Instance $instance_id not found or has no IP"
        exit 1
    fi

    echo "Connecting to $ip..."
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$ip" "$@"
}

cmd_logs() {
    check_api_key

    if [ ! -f "$INSTANCE_FILE" ]; then
        echo "No tracked instance."
        exit 1
    fi

    local instance_id=$(cat "$INSTANCE_FILE")
    local ip=$(get_instance_ip "$instance_id")

    if [ -z "$ip" ]; then
        echo "Instance $instance_id not found or has no IP"
        exit 1
    fi

    echo "Tailing logs from $ip... (Ctrl+C to stop)"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$ip" "tail -f $REMOTE_DIR/train.log"
}

cmd_stop() {
    check_api_key

    if [ ! -f "$INSTANCE_FILE" ]; then
        echo "No tracked instance to stop."
        echo "Use './scripts/lambda_launch.sh status' to see running instances."
        exit 1
    fi

    local instance_id=$(cat "$INSTANCE_FILE")

    echo "Terminating instance $instance_id..."
    local response=$(api_call POST "/instance-operations/terminate" "{\"instance_ids\": [\"$instance_id\"]}")

    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
terminated = data.get('data', {}).get('terminated_instances', [])
if terminated:
    print(f'Terminated: {terminated}')
else:
    print('Termination response:', data)
"

    rm -f "$INSTANCE_FILE"
    echo "Done."
}

cmd_setup() {
    local config_file="${1:-config/train_gpt2_from_scratch.py}"
    check_api_key

    if [ ! -f "$INSTANCE_FILE" ]; then
        echo "No tracked instance found. Use 'launch' first."
        exit 1
    fi

    local instance_id=$(cat "$INSTANCE_FILE")
    local ip=$(get_instance_ip "$instance_id")

    if [ -z "$ip" ]; then
        echo "Could not find IP for instance $instance_id"
        exit 1
    fi

    run_setup "$ip" "$config_file"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    launch)
        cmd_launch "${2:-}"
        ;;
    setup)
        cmd_setup "${2:-}"
        ;;
    status)
        cmd_status
        ;;
    ssh)
        cmd_ssh
        ;;
    logs)
        cmd_logs
        ;;
    stop)
        cmd_stop
        ;;
    *)
        echo "Usage: $0 {launch|setup|status|ssh|logs|stop} [config_file]"
        echo ""
        echo "Commands:"
        echo "  launch [config]  - Spin up instance and start training"
        echo "  setup [config]   - Resume setup/training on current instance"
        echo "  status           - Show running instances"
        echo "  ssh              - SSH into the running instance"
        echo "  logs             - Tail training logs"
        echo "  stop             - Terminate the instance"
        echo ""
        echo "Required environment variables:"
        echo "  LAMBDA_API_KEY   - Get from https://cloud.lambdalabs.com/api-keys"
        echo ""
        echo "Optional environment variables:"
        echo "  HF_TOKEN         - HuggingFace token for dataset access"
        echo "  WANDB_API_KEY    - Weights & Biases API key for logging"
        exit 1
        ;;
esac
