from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gpt2 import GPT2
from schemas import MessageSchema

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/")
async def chatbot_response(message: MessageSchema):
    # Convert the user's question into an embedding and query the Vector DB
    # to check if a similar question has been asked before.

    # If a close match is found in the Vector DB, retrieve the associated
    # answer from the FAQ database instead of using the model. Otherwise,
    # proceed to use the model.
    model = GPT2()
    response_message = model.return_response(message.message)

    # Store the embedding of the user's question in the Vector DB, and the
    # answer in FAQ DB, for future reference.
    return {"response": response_message}
