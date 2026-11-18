# Keep the 256-token baseline separate from experiments that include padded tails.
export BIDAW_PAD_TAIL="${BIDAW_PAD_TAIL:-1}"
case "$BIDAW_PAD_TAIL" in
    0) LMCACHE_L2_PATH="${DEMO_DIR}/l2_ssd" ;;
    1) LMCACHE_L2_PATH="${DEMO_DIR}/l2_ssd/padded-tail" ;;
    *) printf '%s\n' 'BIDAW_PAD_TAIL must be 0 or 1.' >&2; return 1 ;;
esac
