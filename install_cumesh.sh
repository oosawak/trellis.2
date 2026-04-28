#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================================="
echo "CuMesh CUDA 12 Build (Pre-Patched)"
echo "=================================================="
echo ""

# Step 1: Clone
echo "[1/5] Cloning CuMesh..."
cd /tmp
rm -rf CuMesh
git clone https://github.com/JeffreyXiang/CuMesh.git --recursive 2>&1 | tail -3
cd CuMesh

echo "    Location: $(pwd)"

# Step 2: Replace with pre-patched file
echo ""
echo "[2/5] Applying CUDA 12.0+ patch..."
cp "$SCRIPT_DIR/clean_up.cu" src/clean_up.cu
echo "    [+] Patched clean_up.cu installed"

# Verify
remaining=$(grep -c 'cuda::std::__4' src/clean_up.cu || true)
if [ "$remaining" -eq 0 ]; then
    echo "    [✓] No cuda::std::__4 references"
else
    echo "    [!] ERROR: Still has cuda::std::__4 references!"
    exit 1
fi

# Step 3: Build
echo ""
echo "[3/5] Compiling CUDA extensions (3-5 minutes)..."
python3 setup.py build_ext --inplace 2>&1 | tee build.log
BUILD_RESULT=$?

if [ $BUILD_RESULT -ne 0 ]; then
    echo ""
    echo "[-] Compilation failed! Last 50 lines:"
    tail -50 build.log
    exit 1
fi
echo "    [+] Compilation successful"

# Step 4: Install
echo ""
echo "[4/5] Installing Python package..."
pip install -e . --no-build-isolation 2>&1 | tail -5
INSTALL_RESULT=$?

if [ $INSTALL_RESULT -ne 0 ]; then
    echo "[-] Installation failed!"
    exit 1
fi
echo "    [+] Installation successful"

# Step 5: Verify
echo ""
echo "[5/5] Verifying installation..."
python3 -c "import cumesh; print('✓ cumesh module imported')" 2>&1
VERIFY_RESULT=$?

if [ $VERIFY_RESULT -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ SUCCESS! CuMesh is ready"
    echo "=================================================="
    echo ""
    echo "Next: Run TRELLIS setup"
    echo "  cd ~/TRELLIS.2"
    echo "  ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --o-voxel --flexgemm"
else
    echo "[-] Import verification failed!"
    exit 1
fi
