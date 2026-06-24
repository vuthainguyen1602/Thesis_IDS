#!/usr/bin/env bash
# =============================================================================
# setup_swap_jetson.sh — create an NVMe-backed swapfile on each Jetson worker
# so the Spark driver (3g) + executor (3g) on Jetson #1 cannot OOM the 8GB RAM
# during heavy training (XGBoost, RF, MLP).
#
# Target: NVIDIA Jetson Orin Nano Super Developer Kit (8GB LPDDR5, 256GB NVMe).
# Run ONCE on EACH Jetson:   sudo ./setup_swap_jetson.sh
# Override size:             sudo SWAP_GB=8 ./setup_swap_jetson.sh
# =============================================================================
set -euo pipefail

SWAP_GB="${SWAP_GB:-8}"            # default 8GB (>= the 4GB minimum recommended)
SWAPFILE="${SWAPFILE:-/swapfile}"  # lives on the 256GB NVMe root (the SSD)
SWAPPINESS="${SWAPPINESS:-10}"     # low: prefer RAM, use swap only under pressure

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (sudo $0)"; exit 1
fi

echo ">> Target swapfile: ${SWAPFILE}  size: ${SWAP_GB}G  swappiness: ${SWAPPINESS}"

# --- 0. Confirm the swapfile path is on NVMe/SSD (not the SD card) -----------
DEV="$(findmnt -n -o SOURCE --target "$(dirname "${SWAPFILE}")" || true)"
echo ">> Filesystem backing $(dirname "${SWAPFILE}") = ${DEV:-unknown}"
case "${DEV}" in
  *nvme*|*sda*) echo "   OK: NVMe/SSD-backed." ;;
  *mmcblk*)     echo "   WARNING: this looks like an SD card. Point SWAPFILE= to the NVMe mount for better endurance/speed." ;;
  *)            echo "   NOTE: could not auto-detect device type; continuing." ;;
esac

# --- 1. Disable JetPack zram (compressed-RAM swap) ---------------------------
# zram steals from the same 8GB RAM we are trying to protect; a real NVMe
# swapfile is what gives true headroom for Spark heaps.
if systemctl list-unit-files 2>/dev/null | grep -q '^nvzramconfig'; then
  echo ">> Disabling nvzramconfig (JetPack zram)..."
  systemctl disable --now nvzramconfig 2>/dev/null || true
fi
for z in /dev/zram*; do
  [[ -e "$z" ]] && swapoff "$z" 2>/dev/null || true
done

# --- 2. Remove any previous instance of this swapfile ------------------------
if swapon --show=NAME --noheadings 2>/dev/null | grep -qx "${SWAPFILE}"; then
  echo ">> Turning off existing ${SWAPFILE}..."
  swapoff "${SWAPFILE}"
fi
[[ -f "${SWAPFILE}" ]] && rm -f "${SWAPFILE}"

# --- 3. Allocate the swapfile ------------------------------------------------
echo ">> Allocating ${SWAP_GB}G..."
if ! fallocate -l "${SWAP_GB}G" "${SWAPFILE}" 2>/dev/null; then
  # fallocate can fail on some FS (e.g. ext4 with certain options); fall back to dd
  dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$((SWAP_GB*1024)) status=progress
fi
chmod 600 "${SWAPFILE}"
mkswap "${SWAPFILE}"
swapon "${SWAPFILE}"

# --- 4. Persist across reboots ----------------------------------------------
if ! grep -qE "^[^#]*[[:space:]]${SWAPFILE}[[:space:]]|^${SWAPFILE}[[:space:]]" /etc/fstab; then
  echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
  echo ">> Added ${SWAPFILE} to /etc/fstab"
fi

# --- 5. Tune swappiness (persisted) -----------------------------------------
sysctl -w vm.swappiness="${SWAPPINESS}" >/dev/null
mkdir -p /etc/sysctl.d
echo "vm.swappiness=${SWAPPINESS}" > /etc/sysctl.d/99-ids-swap.conf

# --- 6. Report ---------------------------------------------------------------
echo ">> DONE. Current memory + swap:"
free -h
echo
echo ">> swapon summary:"
swapon --show
