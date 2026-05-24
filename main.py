"""Entry point for AI Agent Company API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=52638, reload=True)
