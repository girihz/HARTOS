#!/bin/bash
# pip install --upgrade pip
# docker build -t langchain_gpt:v1 .
# docker kill langchain
# docker rm langchain
# docker run -dp 5055:5000 --network host --name langchain langchain_gpt:v1

    # ── Central-only: master key + signed manifest ──
    if [ "${TIER}" = "central" ]; then
        RUN_ARGS="${RUN_ARGS} -e HEVOLVE_ENFORCEMENT_MODE=hard"
        RUN_ARGS="${RUN_ARGS} -e HEVOLVE_DEV_MODE=false"

        if [ -f "${MASTER_KEY_FILE}" ]; then
            MASTER_KEY_VAL="$(sudo cat "${MASTER_KEY_FILE}" 2>/dev/null || cat "${MASTER_KEY_FILE}" 2>/dev/null)"
            if [ -n "${MASTER_KEY_VAL}" ]; then
                RUN_ARGS="${RUN_ARGS} -e HEVOLVE_MASTER_PRIVATE_KEY=${MASTER_KEY_VAL}"
                info "Master key loaded"
            fi
        else
            warn "Master key not found at ${MASTER_KEY_FILE}"
        fi

        if [ -f "${MANIFEST}" ]; then
            RUN_ARGS="${RUN_ARGS} -v ${MANIFEST}:/app/release_manifest.json:ro"
            info "Release manifest mounted"
        else
            warn "Release manifest not found at ${MANIFEST}"
        fi
    fi

    # Volume mounts (all tiers)
    RUN_ARGS="${RUN_ARGS} -v ${LOG_DIR}:/app/logs"
    RUN_ARGS="${RUN_ARGS} -v ${IMAGE_DIR}:/app/output_images"

    info "Tier: ${TIER}"
    info "Starting ${CONTAINER_NAME} on port ${PORT}..."
    ${DOCKER_CMD} run ${RUN_ARGS} "${IMAGE}"

    # Wait for startup and show status
    sleep 2
    if ${DOCKER_CMD} ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "Container running"
        echo ""
        do_status
    else
        error "Container failed to start. Check logs:"
        ${DOCKER_CMD} logs "${CONTAINER_NAME}" --tail 30
        exit 1
    fi
}

# ── Logs ─────────────────────────────────────────────────────
do_logs() {
    ${DOCKER_CMD} logs "${CONTAINER_NAME}" -f --tail 100
}

# ── Status ───────────────────────────────────────────────────
do_status() {
    if ${DOCKER_CMD} ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "  Container:  ${CONTAINER_NAME}"
        echo "  Image:      ${IMAGE}"
        echo "  Tier:       ${TIER}"
        echo "  Port:       ${PORT}"
        echo "  Status:     $(${DOCKER_CMD} ps --format '{{.Status}}' -f name=${CONTAINER_NAME})"
        echo ""

        # Quick health check
        if command -v curl > /dev/null 2>&1; then
            HEALTH=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/status" 2>/dev/null || echo "000")
            if [ "${HEALTH}" = "200" ]; then
                info "Health check: OK (HTTP 200)"
            else
                warn "Health check: HTTP ${HEALTH} (container may still be starting)"
            fi
        fi
    else
        warn "Container ${CONTAINER_NAME} is not running"
    fi
}

# ── Main ─────────────────────────────────────────────────────
case "${1:-}" in
    build)
        do_build
        ;;
    run)
        do_run
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        do_run
        ;;
    logs)
        do_logs
        ;;
    status)
        do_status
        ;;
    ""|start)
        do_build
        do_run
        ;;
    *)
        echo "Usage: scripts/start_docker.sh [build|run|stop|restart|logs|status] [--tier central|regional|flat]"
        echo ""
        echo "Tiers:"
        echo "  central   — Production server with master key + cloud DB"
        echo "  regional  — Regional LLM host + federation"
        echo "  flat      — Local/desktop (default)"
        exit 1
        ;;
esac
