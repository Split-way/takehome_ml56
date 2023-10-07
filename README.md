# Launching the project

1. Clone the repository:
```
git clone https://github.com/Split-way/coding_challenge_ml56.git
```
2. Run docker-compose.yaml:
```
docker compose up
```
* or if you have older version of docker:
```
docker-compose up
```
3. Open `http://localhost:8080/` in your browser; there you can chat with the model.

## Additional requirement
### How would I containerize this application for production

I'd use Kubernetes on VMs optimized for LLMs (with specific types of GPUs and a significant amount of RAM) and a bigger LLM, like LLaMa 2. I'd refactor the frontend to React or Svelte, run multiple pods of the backend and frontend, and load balance between them.