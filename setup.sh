#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please run: source ./setup.sh"
    exit 1
fi

EDGE_TOOLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_TOOLS_CLI_DIR="$EDGE_TOOLS_ROOT/swift/CLI"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

edge_tools_setup() {
    local configuration="${EDGE_CONFIGURATION:-release}"

    echo "Setting up Swift Edge Tools..."
    echo "=============================="
    echo ""

    if ! command -v swift &> /dev/null; then
        echo -e "${RED}Error: swift is not installed.${NC}"
        echo "  macOS: install Xcode, or https://swift.org/install"
        return 1
    fi
    echo -e "${GREEN}✓ $(swift --version 2>&1 | head -1)${NC}"

    echo ""
    echo -e "${BLUE}Step 1: Fetching submodules...${NC}"
    if ! git -C "$EDGE_TOOLS_ROOT" submodule update --init --recursive; then
        echo -e "${RED}Failed to fetch submodules.${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Submodules ready${NC}"

    echo ""
    echo -e "${BLUE}Step 2: Building the edge CLI ($configuration)...${NC}"
    echo "This compiles XGrammar and MLX, so the first build takes a few minutes."
    if ! swift build --package-path "$EDGE_TOOLS_CLI_DIR" -c "$configuration"; then
        echo -e "${RED}Failed to build the edge CLI.${NC}"
        return 1
    fi

    local bin_path
    bin_path="$(swift build --package-path "$EDGE_TOOLS_CLI_DIR" -c "$configuration" --show-bin-path)"
    echo -e "${GREEN}✓ Built $bin_path/edge${NC}"

    export PATH="$bin_path:$PATH"

    echo ""
    echo -e "${GREEN}Ready. 'edge' is on your PATH for this shell.${NC}"
    echo ""
    echo "  edge Cactus-Compute/needle -p \"Set a timer for 20 minutes\" --tools tools.json"
    echo "  edge info Cactus-Compute/needle"
    echo "  edge bench Cactus-Compute/needle -p \"...\" --repeat-count 20"
    echo ""
    echo "Models are cached in \${HF_HOME:-~/.cache/huggingface}."
}

edge_tools_setup
unset -f edge_tools_setup
