#!/bin/bash


API_CHAT_URL="http://localhost:3000/chat"
API_ROLLBACK_URL="http://localhost:3000/rollback"
SESSION_ID="test-rollback-session-$(date +%s)"  


GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

print_step() {
    echo -e "${BLUE}[STEP] $1${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}


send_chat_message() {
    local message=$1
    print_step "Sending message: \"$message\""
    
    response=$(curl -s -X POST $API_CHAT_URL \
         -H "Content-Type: application/json" \
         -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"$message\"}")
    

    model_response=$(echo $response | sed 's/.*"response":"\(.*\)"}/\1/' | sed 's/\\n/\n/g')
    
    print_info "Model response:"
    echo -e "$model_response"
    echo "-------------------"
}


do_rollback() {
    local target_turn=$1
    
    if [ -z "$target_turn" ]; then
        print_step "Performing rollback to previous snapshot"
        rollback_data="{\"session_id\":\"$SESSION_ID\"}"
    else
        print_step "Performing rollback to turn $target_turn"
        rollback_data="{\"session_id\":\"$SESSION_ID\",\"rollback_to_turn\":$target_turn}"
    fi
    
    rollback_response=$(curl -s -X POST $API_ROLLBACK_URL \
         -H "Content-Type: application/json" \
         -d "$rollback_data")
    

    success=$(echo $rollback_response | grep -o '"success":true' | wc -l)
    message=$(echo $rollback_response | sed 's/.*"message":"\([^"]*\)".*/\1/')
    current_turn=$(echo $rollback_response | sed 's/.*"current_turn":\([0-9]*\).*/\1/')
    
    if [ "$success" -eq 1 ]; then
        print_info "Rollback successful: $message"
        print_info "Current turn: $current_turn"
    else
        print_error "Rollback failed: $message"
        print_info "Current turn: $current_turn"
    fi
    
    echo "-------------------"
}


print_info "Starting KVCache rollback test"
print_info "Session ID: $SESSION_ID"
echo "===================="


send_chat_message "Hello, please introduce yourself"


send_chat_message "Let's move to Paris. I like France."


send_chat_message "What can you do?"


print_info "Testing rollback to specific turn"
do_rollback 1


send_chat_message "What can you do?"


print_info "Testing rollback to previous snapshot"
do_rollback


print_info "Testing rollback to non-existent turn (should fail)"
do_rollback 10


echo "===================="
print_info "Test complete" 



# curl -s -X POST http://localhost:3000/rollback \
#      -H "Content-Type: application/json" \
#      -d '{"session_id":"session1","rollback_to_turn":2}'
