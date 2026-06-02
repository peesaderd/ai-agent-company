module.exports = {
  apps: [{
    name: "ai-agent-company",
    cwd: "/home/openhands/ai-agent-company",
    script: "/home/openhands/ai-agent-company/venv/bin/uvicorn",
    args: "api.server:app --host 0.0.0.0 --port 52638",
    exec_interpreter: "none",
    env: {
      LLM_BACKEND: "ollama",
      OLLAMA_MODEL: "qwen2.5:7b",
      OLLAMA_BASE_URL: "http://localhost:11434"
    },
    watch: false,
    autorestart: true,
    max_restarts: 10,
    restart_delay: 5000,
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    error_file: "/home/openhands/ai-agent-company/logs/err.log",
    out_file: "/home/openhands/ai-agent-company/logs/out.log",
    merge_logs: true,
    pid_file: "/home/openhands/ai-agent-company/logs/pid.pid"
  }, {
    name: "openclaw-gateway",
    cwd: "/home/openhands",
    script: "/home/openhands/openclaw-gw.sh",
    exec_interpreter: "bash",
    watch: false,
    autorestart: true,
    max_restarts: 10,
    restart_delay: 5000,
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    error_file: "/home/openhands/.pm2/logs/openclaw-gateway-error.log",
    out_file: "/home/openhands/.pm2/logs/openclaw-gateway-out.log",
    merge_logs: false,
    env: {
      OPENAI_API_KEY: "sk-LTP2Z9x9adJjxgzUfcWjoQS9lxekHw5xMhKUs5NkCCULT9jhCryWgCFOPdwfngi0",
      OPENAI_BASE_URL: "https://api.opencode.ai/v1",
      OPENCLAW_DEFAULT_MODEL: "openai/deepseek-v4-flash"
    }
  }]
};
