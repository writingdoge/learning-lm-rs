#!/bin/bash

API_URL="http://localhost:3000/chat"
SESSION_ID="test-session-english2"

MESSAGES=(
  "Hello, how are you?"
  "Could you tell me a short story?"
  "What's the capital of France?"
)

for MSG in "${MESSAGES[@]}"; do
  
  echo "Sending message: $MSG"
  echo "-------------------"

  curl -X POST $API_URL \
       -H "Content-Type: application/json" \
       -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"$MSG\"}"

  echo -e "\n"  
  echo "-------------------"
  echo -e "\n"  
  sleep 1      
done

echo "OK"
