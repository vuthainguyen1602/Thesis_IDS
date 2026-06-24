#!/bin/bash
# Resolve SPARK_HOME for pip-installed PySpark.

resolve_spark_home() {
    local candidate from_pyspark
    from_pyspark="$(python -c "import pyspark; print(pyspark.__path__[0])" 2>/dev/null || true)"
    for candidate in "${SPARK_HOME:-}" "$from_pyspark"; do
        [ -z "$candidate" ] && continue
        if [ -f "$candidate/sbin/start-worker.sh" ] \
            || [ -f "$candidate/sbin/start-master.sh" ] \
            || [ -f "$candidate/sbin/start-slave.sh" ] \
            || [ -f "$candidate/bin/spark-class" ]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "[ERR] Spark not found (need bin/spark-class or sbin/start-worker.sh)." >&2
    echo "      SPARK_HOME=${SPARK_HOME:-<unset>}" >&2
    echo "      pyspark path=${from_pyspark:-<not installed>}" >&2
    if [ -n "$from_pyspark" ]; then
        echo "      ls ${from_pyspark}/bin:" >&2
        ls -la "${from_pyspark}/bin" 2>&1 | head -5 >&2 || true
    fi
    echo "      On Jetson run:" >&2
    echo "        cd ~/Thesis_IDS/jetson && source venv/bin/activate" >&2
    echo "        pip install --force-reinstall 'pyspark==3.4.1'" >&2
    return 1
}

export_spark_home() {
    local home
    home="$(resolve_spark_home)" || exit 1
    export SPARK_HOME="$home"
}

start_spark_worker() {
    local master_url="$1"
    local cores="$2"
    local memory="$3"

    unset SPARK_MASTER
    if [ -f "$SPARK_HOME/sbin/start-worker.sh" ]; then
        "$SPARK_HOME/sbin/start-worker.sh" --cores "$cores" --memory "$memory" "$master_url"
    elif [ -f "$SPARK_HOME/sbin/start-slave.sh" ]; then
        "$SPARK_HOME/sbin/start-slave.sh" --cores "$cores" --memory "$memory" "$master_url"
    elif [ -f "$SPARK_HOME/bin/spark-class" ]; then
        "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
            --cores "$cores" --memory "$memory" "$master_url"
    else
        echo "[ERR] No worker launcher under $SPARK_HOME" >&2
        exit 1
    fi
}
