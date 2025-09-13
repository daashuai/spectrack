set -x
# 解析 debug 参数
if [ "$1" == "--debug" ]; then
  DEBUG=1
  PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
  export DEBUG=$DEBUG
  export DEBUG_PORT=$PORT
  echo "[run train_ppo_ray] Using port: $PORT"
else
  DEBUG=0
  PORT=5678
fi
PROJECT_DIR=$(pwd)
echo "DEBUG=$DEBUG" > "$PROJECT_DIR/port.txt"
echo "DEBUG_PORT=$PORT" >> "$PROJECT_DIR/port.txt"
echo "[debug] Wrote DEBUG=$DEBUG and DEBUG_PORT=$PORT to $PROJECT_DIR/port.txt"

python read_spec_by_file.py ../spec8k/test/八涧堡路口西3倍.TXT
